// acl_bench: acl_profile with timings, for a same-host speed comparison with ACL.
// On top of acl_profile's output it reports
//   parse_seconds                BVH read + parse + ACL track build
//   compress_seconds_min / _median   over 1 + --reps compressions (--stat=none: without ACL's stats writer)
//   decompress_seconds           full-clip decompression (seek + decompress_tracks at every frame), debug settings
//   decompress_default_seconds   the same with ACL's production runtime settings (default_transform_decompression_settings)
//   pose_random_ns / pose_random_default_ns   one pose sampled at a uniformly random time with interpolation
//   error_seconds                the per-bone / per-frame shell-error pass (--no-error skips it)
// Compressed buffers and acl_profile's output fields are identical to acl_profile's.
//
// acl_profile: load a BVH clip, convert it to ACL transform tracks, compress it with
// configurable settings and dump a detailed stage-by-stage size / error profile (sjson).
//
// Usage: acl_profile <input.bvh> <output.sjson> [options]
//   --level=automatic|lowest|low|medium|high|highest   (default: automatic, as ACL default settings)
//   --rot=drop_w_variable|drop_w_full|full             (default: drop_w_variable)
//   --vec=variable|full                                (default: variable)
//   --bind-default          use the rest pose (BVH OFFSET, identity rotation) as the default sub-track value
//   --no-loop-opt           disable optimize_loops
//   --no-strip-trivial      disable trivial keyframe stripping
//   --strip-threshold=X     keyframe stripping error threshold (cm)
//   --strip-proportion=X    minimum keyframe stripping proportion [0,1]
//   --precision=X           precision threshold in cm (default 0.01)
//   --shell=X               shell distance in cm (default 3.0)
//   --scale=auto|<float>    unit scale applied to translations (auto: normalize to cm if the rig looks non-cm)
//   --stat=summary|detailed|exhaustive  (default detailed)
//   --dump-fk=<path>        dump raw & lossy object-space joint positions of the first frames (validation)
//   --export-local=<path>   export raw local qvv as float32 binary [frames][bones][7] (qx qy qz qw tx ty tz)
//   --dump-quantized=<path> dump v2: constants, clip ranges, per segment ranges + quantized integers (binary)
//   --export-lossy=<path>   export ACL-decompressed local qvv as float32 [frames][bones][7] (same layout as --export-local)
//   --dump-compressed=<path> write the raw compressed_tracks buffer
//   --no-bitrates           do not dump per-segment/per-bone bit rates

#include <sjson/writer.h>

#include <acl/core/ansi_allocator.h>
#include <acl/core/floating_point_control.h>
#include <acl/core/compressed_tracks.h>
#include <acl/core/impl/compressed_headers.h>
#include <acl/core/impl/debug_track_writer.h>
#include <acl/compression/compress.h>
#include <acl/compression/track_array.h>
#include <acl/compression/track_error.h>
#include <acl/compression/transform_error_metrics.h>
#include <acl/decompression/decompress.h>

#include <rtm/quatf.h>
#include <rtm/qvvf.h>
#include <rtm/vector4f.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace acl;

namespace
{
	//////////////////////////////////////////////////////////////////////////
	// BVH
	//////////////////////////////////////////////////////////////////////////
	struct bvh_joint
	{
		std::string name;
		int parent = -1;
		float offset[3] = { 0.0F, 0.0F, 0.0F };
		std::vector<std::string> channels;
		bool is_end_site = false;
		int channel_offset = 0;
	};

	struct bvh_clip
	{
		std::vector<bvh_joint> joints;
		int num_frames = 0;
		float frame_time = 0.0F;
		int num_channels = 0;
		std::vector<float> motion;	// num_frames * num_channels
	};

	static bool read_file(const char* path, std::string& out)
	{
		std::FILE* f = std::fopen(path, "rb");
		if (!f) return false;
		std::fseek(f, 0, SEEK_END);
		const long size = std::ftell(f);
		std::fseek(f, 0, SEEK_SET);
		out.resize(size_t(size));
		const size_t read = std::fread(&out[0], 1, size_t(size), f);
		std::fclose(f);
		return read == size_t(size);
	}

	static bool parse_bvh(const std::string& text, bvh_clip& clip, std::string& error)
	{
		// Tokenize the hierarchy part manually, parse MOTION numbers with strtof for speed
		size_t pos = 0;
		const size_t len = text.size();
		auto skip_ws = [&]() { while (pos < len && std::isspace(static_cast<unsigned char>(text[pos]))) pos++; };
		auto next_token = [&]() -> std::string
		{
			skip_ws();
			const size_t start = pos;
			while (pos < len && !std::isspace(static_cast<unsigned char>(text[pos]))) pos++;
			return text.substr(start, pos - start);
		};
		// Joint names may contain spaces (e.g. "JOINT U3DMesh 1"): take the rest of the line
		auto next_name = [&]() -> std::string
		{
			skip_ws();
			const size_t start = pos;
			while (pos < len && text[pos] != '\n' && text[pos] != '\r') pos++;
			size_t end = pos;
			while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) end--;
			return text.substr(start, end - start);
		};

		std::string tok = next_token();
		if (tok != "HIERARCHY") { error = "Expected HIERARCHY"; return false; }

		std::vector<int> stack;
		int channel_count = 0;
		while (true)
		{
			tok = next_token();
			if (tok.empty()) { error = "Unexpected EOF in hierarchy"; return false; }
			if (tok == "MOTION") break;

			if (tok == "ROOT" || tok == "JOINT")
			{
				bvh_joint j;
				j.name = next_name();
				j.parent = stack.empty() ? -1 : stack.back();
				clip.joints.push_back(j);
				stack.push_back(int(clip.joints.size()) - 1);
			}
			else if (tok == "End")
			{
				next_token();	// Site
				bvh_joint j;
				j.name = clip.joints[size_t(stack.back())].name + "_end";
				j.parent = stack.back();
				j.is_end_site = true;
				clip.joints.push_back(j);
				stack.push_back(int(clip.joints.size()) - 1);
			}
			else if (tok == "{")
			{
			}
			else if (tok == "}")
			{
				if (stack.empty()) { error = "Unbalanced }"; return false; }
				stack.pop_back();
			}
			else if (tok == "OFFSET")
			{
				bvh_joint& j = clip.joints[size_t(stack.back())];
				for (int i = 0; i < 3; ++i) j.offset[i] = std::strtof(next_token().c_str(), nullptr);
			}
			else if (tok == "CHANNELS")
			{
				bvh_joint& j = clip.joints[size_t(stack.back())];
				const int n = std::atoi(next_token().c_str());
				j.channel_offset = channel_count;
				for (int i = 0; i < n; ++i) j.channels.push_back(next_token());
				channel_count += n;
			}
			else
			{
				error = "Unexpected token in hierarchy: " + tok;
				return false;
			}
		}

		clip.num_channels = channel_count;

		tok = next_token();	// Frames:
		if (tok.rfind("Frames", 0) != 0) { error = "Expected Frames:"; return false; }
		if (tok == "Frames") next_token();	// ':' separated
		clip.num_frames = std::atoi(next_token().c_str());
		tok = next_token();	// Frame
		tok = next_token();	// Time:
		if (tok == "Time") next_token();
		clip.frame_time = std::strtof(next_token().c_str(), nullptr);

		clip.motion.resize(size_t(clip.num_frames) * size_t(clip.num_channels));
		const char* p = text.c_str() + pos;
		char* end = nullptr;
		for (size_t i = 0; i < clip.motion.size(); ++i)
		{
			const float v = std::strtof(p, &end);
			if (end == p) { error = "Not enough motion values"; return false; }
			clip.motion[i] = v;
			p = end;
		}
		return true;
	}

	//////////////////////////////////////////////////////////////////////////
	// Options
	//////////////////////////////////////////////////////////////////////////
	struct options
	{
		const char* input = nullptr;
		const char* output = nullptr;
		compression_level8 level = compression_level8::automatic;
		rotation_format8 rotation_format = rotation_format8::quatf_drop_w_variable;
		vector_format8 vector_format = vector_format8::vector3f_variable;
		bool bind_default = false;
		bool loop_opt = true;
		bool strip_trivial = true;
		float strip_threshold = 0.0F;
		float strip_proportion = 0.0F;
		float precision = 0.01F;
		float shell = 3.0F;
		float scale = -1.0F;	// auto
		stat_logging logging = stat_logging::detailed;
		const char* dump_fk = nullptr;
		const char* export_local = nullptr;
		const char* dump_quantized = nullptr;
		const char* export_lossy = nullptr;
		const char* dump_compressed = nullptr;
		bool dump_bitrates = true;
		uint32_t reps = 0;			// extra timed compressions (logging none)
		uint32_t random_poses = 20000;	// random-time pose samples per settings
		bool no_error = false;		// skip the error pass
	};

	static bool starts_with(const char* s, const char* prefix, const char*& rest)
	{
		const size_t n = std::strlen(prefix);
		if (std::strncmp(s, prefix, n) == 0) { rest = s + n; return true; }
		return false;
	}

	static bool parse_options(int argc, char** argv, options& o)
	{
		if (argc < 3) return false;
		o.input = argv[1];
		o.output = argv[2];
		for (int i = 3; i < argc; ++i)
		{
			const char* a = argv[i];
			const char* v = nullptr;
			if (starts_with(a, "--level=", v)) { if (!get_compression_level(v, o.level)) { printf("bad level %s\n", v); return false; } }
			else if (starts_with(a, "--rot=", v))
			{
				if (std::strcmp(v, "drop_w_variable") == 0) o.rotation_format = rotation_format8::quatf_drop_w_variable;
				else if (std::strcmp(v, "drop_w_full") == 0) o.rotation_format = rotation_format8::quatf_drop_w_full;
				else if (std::strcmp(v, "full") == 0) o.rotation_format = rotation_format8::quatf_full;
				else { printf("bad rot %s\n", v); return false; }
			}
			else if (starts_with(a, "--vec=", v))
			{
				if (std::strcmp(v, "variable") == 0) o.vector_format = vector_format8::vector3f_variable;
				else if (std::strcmp(v, "full") == 0) o.vector_format = vector_format8::vector3f_full;
				else { printf("bad vec %s\n", v); return false; }
			}
			else if (std::strcmp(a, "--bind-default") == 0) o.bind_default = true;
			else if (std::strcmp(a, "--no-loop-opt") == 0) o.loop_opt = false;
			else if (std::strcmp(a, "--no-strip-trivial") == 0) o.strip_trivial = false;
			else if (starts_with(a, "--strip-threshold=", v)) o.strip_threshold = std::strtof(v, nullptr);
			else if (starts_with(a, "--strip-proportion=", v)) o.strip_proportion = std::strtof(v, nullptr);
			else if (starts_with(a, "--precision=", v)) o.precision = std::strtof(v, nullptr);
			else if (starts_with(a, "--shell=", v)) o.shell = std::strtof(v, nullptr);
			else if (starts_with(a, "--scale=", v)) { o.scale = (std::strcmp(v, "auto") == 0) ? -1.0F : std::strtof(v, nullptr); }
			else if (starts_with(a, "--stat=", v))
			{
				if (std::strcmp(v, "none") == 0) o.logging = stat_logging::none;
				else if (std::strcmp(v, "summary") == 0) o.logging = stat_logging::summary;
				else if (std::strcmp(v, "detailed") == 0) o.logging = stat_logging::detailed;
				else if (std::strcmp(v, "exhaustive") == 0) o.logging = stat_logging::exhaustive;
				else { printf("bad stat %s\n", v); return false; }
			}
			else if (starts_with(a, "--dump-fk=", v)) o.dump_fk = v;
			else if (starts_with(a, "--export-local=", v)) o.export_local = v;
			else if (starts_with(a, "--dump-quantized=", v)) o.dump_quantized = v;
			else if (starts_with(a, "--export-lossy=", v)) o.export_lossy = v;
			else if (starts_with(a, "--dump-compressed=", v)) o.dump_compressed = v;
			else if (std::strcmp(a, "--no-bitrates") == 0) o.dump_bitrates = false;
			else if (starts_with(a, "--reps=", v)) o.reps = uint32_t(std::strtoul(v, nullptr, 10));
			else if (starts_with(a, "--random-poses=", v)) o.random_poses = uint32_t(std::strtoul(v, nullptr, 10));
			else if (std::strcmp(a, "--no-error") == 0) o.no_error = true;
			else { printf("unknown option %s\n", a); return false; }
		}
		return true;
	}

	//////////////////////////////////////////////////////////////////////////
	// Rig helpers
	//////////////////////////////////////////////////////////////////////////
	static float max_chain_length(const bvh_clip& clip)
	{
		// Longest rest-pose root->leaf path (sum of offset norms, root offset excluded)
		std::vector<float> depth(clip.joints.size(), 0.0F);
		float best = 0.0F;
		for (size_t i = 0; i < clip.joints.size(); ++i)
		{
			const bvh_joint& j = clip.joints[i];
			const float l = std::sqrt(j.offset[0] * j.offset[0] + j.offset[1] * j.offset[1] + j.offset[2] * j.offset[2]);
			depth[i] = (j.parent < 0) ? 0.0F : depth[size_t(j.parent)] + l;
			best = std::max(best, depth[i]);
		}
		return best;
	}

	static rtm::quatf RTM_SIMD_CALL channel_quat(const std::string& channel, float degrees)
	{
		const float rad = degrees * (3.14159265358979323846F / 180.0F);
		rtm::vector4f axis;
		if (channel[0] == 'X') axis = rtm::vector_set(1.0F, 0.0F, 0.0F, 0.0F);
		else if (channel[0] == 'Y') axis = rtm::vector_set(0.0F, 1.0F, 0.0F, 0.0F);
		else axis = rtm::vector_set(0.0F, 0.0F, 1.0F, 0.0F);
		return rtm::quat_from_axis_angle(axis, rad);
	}

	struct rig_info
	{
		std::vector<int> track_joint;		// track index -> joint index
		std::vector<int> joint_track;		// joint index -> track index (-1 for end sites)
		std::vector<int> track_parent;		// track index -> parent track index (-1 for root)
	};

	static void build_rig(const bvh_clip& clip, rig_info& rig)
	{
		rig.joint_track.assign(clip.joints.size(), -1);
		for (size_t i = 0; i < clip.joints.size(); ++i)
		{
			if (clip.joints[i].is_end_site) continue;
			rig.joint_track[i] = int(rig.track_joint.size());
			rig.track_joint.push_back(int(i));
		}
		rig.track_parent.resize(rig.track_joint.size());
		for (size_t t = 0; t < rig.track_joint.size(); ++t)
		{
			const int parent_joint = clip.joints[size_t(rig.track_joint[t])].parent;
			rig.track_parent[t] = parent_joint < 0 ? -1 : rig.joint_track[size_t(parent_joint)];
		}
	}

	// Evaluate the local transform of a track at a frame
	static rtm::qvvf RTM_SIMD_CALL sample_local(const bvh_clip& clip, const bvh_joint& j, int frame, float scale)
	{
		const float* row = clip.motion.data() + size_t(frame) * size_t(clip.num_channels) + size_t(j.channel_offset);
		rtm::quatf q = rtm::quat_identity();
		float t[3] = { j.offset[0], j.offset[1], j.offset[2] };
		for (size_t c = 0; c < j.channels.size(); ++c)
		{
			const std::string& ch = j.channels[c];
			const float v = row[c];
			if (ch.size() >= 9 && ch.compare(1, 8, "rotation") == 0)
			{
				// BVH: listed channels multiply left to right (v' = R1 R2 R3 v): the last listed rotation applies first.
				// RTM: quat_mul(a, b) applies a then b.
				q = rtm::quat_mul(channel_quat(ch, v), q);
			}
			else if (ch.size() >= 9 && ch.compare(1, 8, "position") == 0)
			{
				if (ch[0] == 'X') t[0] += v;
				else if (ch[0] == 'Y') t[1] += v;
				else t[2] += v;
			}
		}
		q = rtm::quat_normalize(q);
		return rtm::qvv_set(q, rtm::vector_set(t[0] * scale, t[1] * scale, t[2] * scale, 0.0F), rtm::vector_set(1.0F));
	}

	//////////////////////////////////////////////////////////////////////////
	// Shell error (same definition as acl::qvvf_transform_error_metric, no scale)
	//////////////////////////////////////////////////////////////////////////
	static float RTM_SIMD_CALL shell_error(const rtm::qvvf& raw, const rtm::qvvf& lossy, float shell)
	{
		const rtm::vector4f vx = rtm::vector_set(shell, 0.0F, 0.0F, 0.0F);
		const rtm::vector4f vy = rtm::vector_set(0.0F, shell, 0.0F, 0.0F);
		const rtm::vector4f vz = rtm::vector_set(0.0F, 0.0F, shell, 0.0F);
		const float ex = rtm::vector_distance3(rtm::qvv_mul_point3_no_scale(vx, raw), rtm::qvv_mul_point3_no_scale(vx, lossy));
		const float ey = rtm::vector_distance3(rtm::qvv_mul_point3_no_scale(vy, raw), rtm::qvv_mul_point3_no_scale(vy, lossy));
		const float ez = rtm::vector_distance3(rtm::qvv_mul_point3_no_scale(vz, raw), rtm::qvv_mul_point3_no_scale(vz, lossy));
		return std::max(ex, std::max(ey, ez));
	}

	static const char* rotation_format_name(rotation_format8 f)
	{
		switch (f)
		{
		case rotation_format8::quatf_full: return "quatf_full";
		case rotation_format8::quatf_drop_w_full: return "quatf_drop_w_full";
		case rotation_format8::quatf_drop_w_variable: return "quatf_drop_w_variable";
		default: return "?";
		}
	}
	static const char* vector_format_name(vector_format8 f)
	{
		return f == vector_format8::vector3f_full ? "vector3f_full" : "vector3f_variable";
	}
}

int main(int argc, char** argv)
{
	options opt;
	if (!parse_options(argc, argv, opt))
	{
		printf("Usage: acl_profile <input.bvh> <output.sjson> [options]\n");
		return 1;
	}

	const auto t_parse0 = std::chrono::steady_clock::now();
	std::string text;
	if (!read_file(opt.input, text)) { printf("Cannot read %s\n", opt.input); return 1; }

	bvh_clip clip;
	std::string err;
	if (!parse_bvh(text, clip, err)) { printf("BVH parse error: %s\n", err.c_str()); return 1; }
	text.clear(); text.shrink_to_fit();

	rig_info rig;
	build_rig(clip, rig);
	const uint32_t num_tracks = uint32_t(rig.track_joint.size());
	const uint32_t num_frames = uint32_t(clip.num_frames);

	const float chain_len_units = max_chain_length(clip);
	float scale = opt.scale;
	bool assumed_cm = true;
	if (scale <= 0.0F)
	{
		// Datasets that are already in centimeters have a root->leaf chain of roughly 50..250 cm
		if (chain_len_units >= 40.0F && chain_len_units <= 300.0F) scale = 1.0F;
		else { scale = 110.0F / chain_len_units; assumed_cm = false; }
	}

	float sample_rate = 1.0F / clip.frame_time;
	{
		const float rounded = std::round(sample_rate);
		if (std::fabs(rounded - sample_rate) / sample_rate < 0.01F) sample_rate = rounded;
	}

	ansi_allocator allocator;

	//////////////////////////////////////////////////////////////////////////
	// Build ACL tracks
	//////////////////////////////////////////////////////////////////////////
	track_array_qvvf tracks(allocator, num_tracks);
	for (uint32_t t = 0; t < num_tracks; ++t)
	{
		const bvh_joint& j = clip.joints[size_t(rig.track_joint[t])];
		track_desc_transformf desc;
		desc.output_index = t;
		desc.parent_index = rig.track_parent[t] < 0 ? k_invalid_track_index : uint32_t(rig.track_parent[t]);
		desc.precision = opt.precision;
		desc.shell_distance = opt.shell;
		if (opt.bind_default)
			desc.default_value = rtm::qvv_set(rtm::quat_identity(), rtm::vector_set(j.offset[0] * scale, j.offset[1] * scale, j.offset[2] * scale, 0.0F), rtm::vector_set(1.0F));

		track_qvvf track = track_qvvf::make_reserve(desc, allocator, num_frames, sample_rate);
		for (uint32_t f = 0; f < num_frames; ++f)
			track[f] = sample_local(clip, j, int(f), scale);
		tracks[t] = std::move(track);
	}

	if (opt.export_local != nullptr)
	{
		std::FILE* f = std::fopen(opt.export_local, "wb");
		if (f)
		{
			std::vector<float> buf(size_t(num_frames) * num_tracks * 7);
			for (uint32_t fr = 0; fr < num_frames; ++fr)
				for (uint32_t t = 0; t < num_tracks; ++t)
				{
					const rtm::qvvf& x = tracks[t][fr];
					float* o = &buf[(size_t(fr) * num_tracks + t) * 7];
					o[0] = rtm::quat_get_x(x.rotation); o[1] = rtm::quat_get_y(x.rotation); o[2] = rtm::quat_get_z(x.rotation); o[3] = rtm::quat_get_w(x.rotation);
					o[4] = rtm::vector_get_x(x.translation); o[5] = rtm::vector_get_y(x.translation); o[6] = rtm::vector_get_z(x.translation);
				}
			std::fwrite(buf.data(), sizeof(float), buf.size(), f);
			std::fclose(f);
		}
	}

	const double parse_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_parse0).count();

	//////////////////////////////////////////////////////////////////////////
	// Compression settings
	//////////////////////////////////////////////////////////////////////////
	qvvf_transform_error_metric error_metric;
	compression_settings settings = get_default_compression_settings();
	settings.level = opt.level;
	settings.rotation_format = opt.rotation_format;
	settings.translation_format = opt.vector_format;
	settings.scale_format = opt.vector_format;
	settings.optimize_loops = opt.loop_opt;
	settings.keyframe_stripping.strip_trivial = opt.strip_trivial;
	settings.keyframe_stripping.threshold = opt.strip_threshold;
	settings.keyframe_stripping.proportion = opt.strip_proportion;
	settings.error_metric = &error_metric;

	const bool is_variable = is_rotation_format_variable(settings.rotation_format) || is_vector_format_variable(settings.translation_format);
	if (!is_variable)
	{
		// ACL refuses contributing error metadata with raw formats
		settings.keyframe_stripping.strip_trivial = false;
		settings.keyframe_stripping.threshold = 0.0F;
		settings.keyframe_stripping.proportion = 0.0F;
	}

	std::FILE* out = std::fopen(opt.output, "w");
	if (!out) { printf("Cannot write %s\n", opt.output); return 1; }

	sjson::FileStreamWriter stream_writer(out);
	sjson::Writer writer(stream_writer);

	writer["file"] = opt.input;
	writer["num_joints_total"] = uint32_t(clip.joints.size());
	writer["num_bones"] = num_tracks;
	writer["num_samples"] = num_frames;
	writer["sample_rate"] = sample_rate;
	writer["frame_time"] = clip.frame_time;
	writer["duration"] = float(num_frames) / sample_rate;
	writer["chain_len_units"] = chain_len_units;
	writer["scale_to_cm"] = scale;
	writer["assumed_cm"] = assumed_cm;
	writer["bvh_bytes"] = uint64_t(0);	// filled by python (text size)
	writer["raw_size_qvv40"] = uint64_t(num_tracks) * num_frames * 40;	// what ACL reports as raw: quat4 + vec3 + vec3 float32
	writer["raw_size_rot4_trans3"] = uint64_t(num_tracks) * num_frames * 28;
	writer["raw_size_rot3_trans3"] = uint64_t(num_tracks) * num_frames * 24;
	writer["raw_size_bvh_channels_f32"] = uint64_t(clip.num_channels) * num_frames * 4;

	writer["settings"] = [&](sjson::ObjectWriter& w)
	{
		w["level"] = get_compression_level_name(settings.level);
		w["rotation_format"] = rotation_format_name(settings.rotation_format);
		w["translation_format"] = vector_format_name(settings.translation_format);
		w["bind_default"] = opt.bind_default;
		w["optimize_loops"] = settings.optimize_loops;
		w["strip_trivial"] = settings.keyframe_stripping.strip_trivial;
		w["strip_threshold"] = settings.keyframe_stripping.threshold;
		w["strip_proportion"] = settings.keyframe_stripping.proportion;
		w["precision"] = opt.precision;
		w["shell_distance"] = opt.shell;
	};

	writer["bone_names"] = [&](sjson::ArrayWriter& w) { for (uint32_t t = 0; t < num_tracks; ++t) w.push(clip.joints[size_t(rig.track_joint[t])].name.c_str()); };
	writer["bone_parents"] = [&](sjson::ArrayWriter& w) { for (uint32_t t = 0; t < num_tracks; ++t) w.push(int32_t(rig.track_parent[t])); };
	writer["bone_num_channels"] = [&](sjson::ArrayWriter& w) { for (uint32_t t = 0; t < num_tracks; ++t) w.push(uint32_t(clip.joints[size_t(rig.track_joint[t])].channels.size())); };

	//////////////////////////////////////////////////////////////////////////
	// Compress
	//////////////////////////////////////////////////////////////////////////
	compressed_tracks* compressed = nullptr;
	error_result result;
	double compress_seconds = 0.0;
	writer["acl"] = [&](sjson::ObjectWriter& w)
	{
		output_stats stats;
		stats.logging = opt.logging;
		stats.writer = &w;
		const auto t0 = std::chrono::steady_clock::now();
		result = compress_track_list(allocator, tracks, settings, compressed, stats);
		compress_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	};

	if (result.any() || compressed == nullptr)
	{
		writer["error"] = result.c_str();
		std::fclose(out);
		printf("Compression failed: %s\n", result.c_str());
		return 1;
	}

	writer["compress_seconds"] = compress_seconds;
	writer["parse_seconds"] = parse_seconds;

	// ---------------- timing: repeated compressions without the stats writer ----------------
	{
		std::vector<double> ct(1, compress_seconds);
		for (uint32_t r = 0; r < opt.reps; ++r)
		{
			output_stats st2;
			st2.logging = stat_logging::none;
			compressed_tracks* c2 = nullptr;
			const auto t0 = std::chrono::steady_clock::now();
			const error_result r2 = compress_track_list(allocator, tracks, settings, c2, st2);
			ct.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
			if (r2.any() || c2 == nullptr || c2->get_size() != compressed->get_size() || std::memcmp(c2, compressed, compressed->get_size()) != 0)
			{
				printf("repeated compression differs\n");
				return 1;
			}
			allocator.deallocate(c2, c2->get_size());
		}
		std::vector<double> s = ct;
		std::sort(s.begin(), s.end());
		writer["compress_seconds_min"] = s.front();
		writer["compress_seconds_median"] = s[s.size() / 2];
		writer["compress_runs"] = uint32_t(s.size());
		writer["compress_logging"] = opt.logging == stat_logging::none ? "none" : "stats";
	}
	// ---------------- end timing ----------------
	writer["compressed_size"] = compressed->get_size();

	//////////////////////////////////////////////////////////////////////////
	// Error: ACL's own (max over tracks/frames with the compression error metric) + our full distribution
	//////////////////////////////////////////////////////////////////////////
	{
		scope_disable_fp_exceptions fp_off;

		decompression_context<debug_transform_decompression_settings> context;
		context.initialize(*compressed);

		// ---------------- timing: ACL's runtime decompression alone (no error metric, no FK) ----------------
		{
			acl_impl::debug_track_writer_variable_defaults timing_writer(allocator, track_type8::qvvf, num_tracks);
			std::vector<rtm::qvvf> tdefaults(num_tracks);
			for (uint32_t t = 0; t < num_tracks; ++t) tdefaults[t] = tracks[t].get_description().default_value;
			timing_writer.default_sub_tracks = tdefaults.data();
			timing_writer.initialize_with_defaults(tracks);
			const float dur_t = compressed->get_duration();
			for (uint32_t f = 0; f < num_frames; ++f)
			{
				context.seek(std::min(float(f) / sample_rate, dur_t), sample_rounding_policy::nearest);
				context.decompress_tracks(timing_writer);
			}
			const uint32_t reps = 10;
			const auto td0 = std::chrono::steady_clock::now();
			for (uint32_t r = 0; r < reps; ++r)
				for (uint32_t f = 0; f < num_frames; ++f)
				{
					context.seek(std::min(float(f) / sample_rate, dur_t), sample_rounding_policy::nearest);
					context.decompress_tracks(timing_writer);
				}
			const double td = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count() / double(reps);
			writer["decompress_seconds"] = td;
			writer["decompress_ns_per_bone_frame"] = td * 1e9 / (double(num_frames) * double(num_tracks));
			writer["decompress_reps"] = int32_t(reps);
		}
		// ---------------- end timing ----------------

		// ---------------- timing: production settings, full clip + random poses ----------------
		{
			acl_impl::debug_track_writer_variable_defaults tw(allocator, track_type8::qvvf, num_tracks);
			std::vector<rtm::qvvf> tdef(num_tracks);
			for (uint32_t t = 0; t < num_tracks; ++t) tdef[t] = tracks[t].get_description().default_value;
			tw.default_sub_tracks = tdef.data();
			tw.initialize_with_defaults(tracks);
			const float dur_t = compressed->get_duration();
			decompression_context<default_transform_decompression_settings> pctx;
			const bool ok = pctx.initialize(*compressed);
			writer["decompress_default_ok"] = ok;
			if (ok)
			{
				auto full_pass = [&]()
				{
					for (uint32_t f = 0; f < num_frames; ++f)
					{
						pctx.seek(std::min(float(f) / sample_rate, dur_t), sample_rounding_policy::nearest);
						pctx.decompress_tracks(tw);
					}
				};
				full_pass();	// warm-up
				uint32_t reps = 0;
				const auto t0 = std::chrono::steady_clock::now();
				double el = 0.0;
				do { full_pass(); ++reps; el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); } while (reps < 10 || el < 0.05);
				writer["decompress_default_seconds"] = el / double(reps);
				writer["decompress_default_reps"] = reps;
				writer["decompress_default_ns_per_bone_frame"] = el / double(reps) * 1e9 / (double(num_frames) * double(num_tracks));
			}
			// random poses: uniform times in [0, duration], interpolated (the runtime sampling mode)
			const uint32_t n_rand = std::max<uint32_t>(1, opt.random_poses);
			std::vector<float> times(n_rand);
			uint64_t lcg = 0x9E3779B97F4A7C15ull;
			for (uint32_t i = 0; i < n_rand; ++i)
			{
				lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
				times[i] = float(double(lcg >> 11) / double(1ull << 53)) * dur_t;
			}
			{
				for (uint32_t i = 0; i < std::min<uint32_t>(n_rand, 256); ++i) { context.seek(times[i], sample_rounding_policy::none); context.decompress_tracks(tw); }
				const auto t0 = std::chrono::steady_clock::now();
				for (uint32_t i = 0; i < n_rand; ++i) { context.seek(times[i], sample_rounding_policy::none); context.decompress_tracks(tw); }
				const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
				writer["pose_random_ns"] = el * 1e9 / double(n_rand);
			}
			if (ok)
			{
				for (uint32_t i = 0; i < std::min<uint32_t>(n_rand, 256); ++i) { pctx.seek(times[i], sample_rounding_policy::none); pctx.decompress_tracks(tw); }
				const auto t0 = std::chrono::steady_clock::now();
				for (uint32_t i = 0; i < n_rand; ++i) { pctx.seek(times[i], sample_rounding_policy::none); pctx.decompress_tracks(tw); }
				const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
				writer["pose_random_default_ns"] = el * 1e9 / double(n_rand);
			}
			writer["pose_random_n"] = n_rand;
		}
		// ---------------- end timing ----------------

		if (!opt.no_error)
		{
		const auto t_err0 = std::chrono::steady_clock::now();

		const track_error e = calculate_compression_error(allocator, tracks, context, error_metric);
		writer["max_error"] = e.error;
		writer["worst_bone"] = e.index;
		writer["worst_time"] = e.sample_time;
		writer["worst_keyframe"] = e.keyframe_index;

		// Full per-bone / per-frame shell error in object space (no scale)
		acl_impl::debug_track_writer_variable_defaults lossy_writer(allocator, track_type8::qvvf, num_tracks);
		std::vector<rtm::qvvf> defaults(num_tracks);
		for (uint32_t t = 0; t < num_tracks; ++t) defaults[t] = tracks[t].get_description().default_value;
		lossy_writer.default_sub_tracks = defaults.data();
		lossy_writer.initialize_with_defaults(tracks);

		std::vector<rtm::qvvf> raw_obj(num_tracks), lossy_obj(num_tracks);
		std::vector<double> bone_sum(num_tracks, 0.0);
		std::vector<float> bone_max(num_tracks, 0.0F);
		std::vector<float> frame_max(num_frames, 0.0F);
		std::vector<float> all_errors;
		all_errors.reserve(size_t(num_tracks) * num_frames);
		double sum = 0.0;
		float max_err = 0.0F;

		std::FILE* fk = opt.dump_fk ? std::fopen(opt.dump_fk, "w") : nullptr;
		const uint32_t fk_frames = std::min<uint32_t>(num_frames, 8);
		std::FILE* lf = opt.export_lossy ? std::fopen(opt.export_lossy, "wb") : nullptr;
		std::vector<float> lossy_row(size_t(num_tracks) * 7);

		const float clip_duration = compressed->get_duration();
		for (uint32_t f = 0; f < num_frames; ++f)
		{
			const float time = std::min(float(f) / sample_rate, clip_duration);
			context.seek(time, sample_rounding_policy::nearest);
			context.decompress_tracks(lossy_writer);

			for (uint32_t t = 0; t < num_tracks; ++t)
			{
				const rtm::qvvf raw_local = tracks[t][f];
				const rtm::qvvf lossy_local = lossy_writer.tracks_typed.qvvf[t];
				if (lf)
				{
					float* o = &lossy_row[size_t(t) * 7];
					o[0] = float(rtm::quat_get_x(lossy_local.rotation)); o[1] = float(rtm::quat_get_y(lossy_local.rotation)); o[2] = float(rtm::quat_get_z(lossy_local.rotation)); o[3] = float(rtm::quat_get_w(lossy_local.rotation));
					o[4] = float(rtm::vector_get_x(lossy_local.translation)); o[5] = float(rtm::vector_get_y(lossy_local.translation)); o[6] = float(rtm::vector_get_z(lossy_local.translation));
				}
				const int p = rig.track_parent[t];
				if (p < 0)
				{
					raw_obj[t] = raw_local;
					lossy_obj[t] = lossy_local;
				}
				else
				{
					raw_obj[t] = rtm::qvv_normalize(rtm::qvv_mul_no_scale(raw_local, raw_obj[size_t(p)]));
					lossy_obj[t] = rtm::qvv_normalize(rtm::qvv_mul_no_scale(lossy_local, lossy_obj[size_t(p)]));
				}
				const float e2 = shell_error(raw_obj[t], lossy_obj[t], opt.shell);
				bone_sum[t] += e2;
				bone_max[t] = std::max(bone_max[t], e2);
				frame_max[f] = std::max(frame_max[f], e2);
				sum += e2;
				max_err = std::max(max_err, e2);
				all_errors.push_back(e2);

				if (fk && f < fk_frames)
				{
					std::fprintf(fk, "%u %u %.6f %.6f %.6f %.6f %.6f %.6f\n", f, t,
						double(rtm::vector_get_x(raw_obj[t].translation)), double(rtm::vector_get_y(raw_obj[t].translation)), double(rtm::vector_get_z(raw_obj[t].translation)),
						double(rtm::vector_get_x(lossy_obj[t].translation)), double(rtm::vector_get_y(lossy_obj[t].translation)), double(rtm::vector_get_z(lossy_obj[t].translation)));
				}
			}
			if (lf) std::fwrite(lossy_row.data(), sizeof(float), lossy_row.size(), lf);
		}
		if (fk) std::fclose(fk);
		if (lf) std::fclose(lf);

		std::sort(all_errors.begin(), all_errors.end());
		auto pct = [&](double p) { return all_errors.empty() ? 0.0F : all_errors[std::min(all_errors.size() - 1, size_t(p * double(all_errors.size())))]; };
		writer["full_error"] = [&](sjson::ObjectWriter& w)
		{
			w["max"] = max_err;
			w["mean"] = all_errors.empty() ? 0.0 : sum / double(all_errors.size());
			w["p50"] = pct(0.50);
			w["p90"] = pct(0.90);
			w["p99"] = pct(0.99);
			w["frac_above_precision"] = all_errors.empty() ? 0.0 : double(all_errors.end() - std::upper_bound(all_errors.begin(), all_errors.end(), opt.precision)) / double(all_errors.size());
			w["bone_max"] = [&](sjson::ArrayWriter& a) { for (uint32_t t = 0; t < num_tracks; ++t) a.push(bone_max[t]); };
			w["bone_mean"] = [&](sjson::ArrayWriter& a) { for (uint32_t t = 0; t < num_tracks; ++t) a.push(float(bone_sum[t] / double(num_frames))); };
		};
		writer["error_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_err0).count();
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Parse the compressed buffer: sub-track types + per segment bit rates
	//////////////////////////////////////////////////////////////////////////
	{
		const acl_impl::tracks_header& hdr = acl_impl::get_tracks_header(*compressed);
		const acl_impl::transform_tracks_header& th = acl_impl::get_transform_tracks_header(*compressed);
		const bool has_scale = hdr.get_has_scale();
		const uint32_t num_out = hdr.num_tracks;
		const uint32_t num_entries = (num_out + 15) / 16;
		const acl_impl::packed_sub_track_types* st = th.get_sub_track_types();
		auto sub_track_type = [&](uint32_t group, uint32_t i) -> uint32_t
		{
			return (st[group * num_entries + i / 16].types >> ((15 - (i % 16)) * 2)) & 3;
		};

		std::vector<uint32_t> rot_types(num_out), trans_types(num_out), scale_types(num_out, 0);
		uint32_t num_anim_rot = 0, num_anim_trans = 0, num_anim_scale = 0;
		for (uint32_t i = 0; i < num_out; ++i)
		{
			rot_types[i] = sub_track_type(0, i);
			trans_types[i] = sub_track_type(1, i);
			if (has_scale) scale_types[i] = sub_track_type(2, i);
			num_anim_rot += rot_types[i] == 2;
			num_anim_trans += trans_types[i] == 2;
			num_anim_scale += scale_types[i] == 2;
		}

		writer["layout"] = [&](sjson::ObjectWriter& w)
		{
			w["has_scale"] = has_scale;
			w["has_stripped_keyframes"] = compressed->has_stripped_keyframes();
			w["looping_wrap"] = compressed->get_looping_policy() == sample_looping_policy::wrap;
			w["num_segments"] = th.num_segments;
			w["num_animated_variable_sub_tracks_padded"] = th.num_animated_variable_sub_tracks;
			w["num_animated_rotation"] = num_anim_rot;
			w["num_animated_translation"] = num_anim_trans;
			w["num_animated_scale"] = num_anim_scale;
			w["num_constant_rotation"] = th.num_constant_rotation_samples;
			w["num_constant_translation"] = th.num_constant_translation_samples;
			w["num_constant_scale"] = th.num_constant_scale_samples;
			w["rotation_types"] = [&](sjson::ArrayWriter& a) { for (uint32_t i = 0; i < num_out; ++i) a.push(rot_types[i]); };
			w["translation_types"] = [&](sjson::ArrayWriter& a) { for (uint32_t i = 0; i < num_out; ++i) a.push(trans_types[i]); };
			// Byte offsets of the major sections (relative to transform_tracks_header)
			w["segment_headers_offset"] = uint32_t(th.segment_headers_offset);
			w["sub_track_types_offset"] = uint32_t(th.sub_track_types_offset);
			w["constant_track_data_offset"] = uint32_t(th.constant_track_data_offset);
			w["clip_range_data_offset"] = uint32_t(th.clip_range_data_offset);
		};

		if (opt.dump_bitrates && is_variable)
		{
			const bool stripped = compressed->has_stripped_keyframes();
			const uint32_t num_segments = th.num_segments;
			const uint32_t* seg_start = num_segments > 1 ? th.get_segment_start_indices() : nullptr;

			if (opt.dump_compressed)
			{
				std::FILE* cf = std::fopen(opt.dump_compressed, "wb");
				if (cf) { std::fwrite(compressed, 1, compressed->get_size(), cf); std::fclose(cf); }
			}

			std::FILE* qf = opt.dump_quantized ? std::fopen(opt.dump_quantized, "wb") : nullptr;
			if (qf)
			{
				// ---- dump v2 header ----
				const uint32_t file_hdr[10] = { 0x32554E51u /* 'QNU2' */, 2u, num_segments, num_out, hdr.num_samples, uint32_t(has_scale),
					uint32_t(settings.rotation_format), uint32_t(settings.translation_format), uint32_t(num_segments > 1), 0u };
				std::fwrite(file_hdr, sizeof(uint32_t), 10, qf);
				for (uint32_t i = 0; i < num_out; ++i) { const uint32_t tt[2] = { rot_types[i], trans_types[i] }; std::fwrite(tt, sizeof(uint32_t), 2, qf); }

				// ---- constants: rotations first (swizzled xxxx yyyy zzzz in groups of 4 for drop-w), then translations (xyz) ----
				{
					std::vector<uint32_t> cbones_r, cbones_t;
					for (uint32_t i = 0; i < num_out; ++i) { if (rot_types[i] == 1) cbones_r.push_back(i); if (trans_types[i] == 1) cbones_t.push_back(i); }
					const uint32_t count = uint32_t(cbones_r.size() + cbones_t.size());
					std::fwrite(&count, sizeof(uint32_t), 1, qf);
					const uint8_t* cd = th.get_constant_track_data();
					const bool swizzle = settings.rotation_format != rotation_format8::quatf_full;
					const uint32_t rot_floats = swizzle ? 3 : 4;
					for (uint32_t k = 0; k < cbones_r.size(); ++k)
					{
						float xyz[3];
						if (swizzle)
						{
							const uint32_t g = k / 4, j = k % 4;
							const uint32_t gsize = std::min<uint32_t>(4, uint32_t(cbones_r.size()) - g * 4);
							const float* base = reinterpret_cast<const float*>(cd + size_t(g) * 4 * 3 * sizeof(float));
							xyz[0] = base[0 * gsize + j]; xyz[1] = base[1 * gsize + j]; xyz[2] = base[2 * gsize + j];
						}
						else
						{
							const float* base = reinterpret_cast<const float*>(cd + size_t(k) * rot_floats * sizeof(float));
							xyz[0] = base[0]; xyz[1] = base[1]; xyz[2] = base[2];
						}
						const uint32_t rec[2] = { cbones_r[k], 0u };
						std::fwrite(rec, sizeof(uint32_t), 2, qf); std::fwrite(xyz, sizeof(float), 3, qf);
					}
					const uint8_t* td = cd + size_t(cbones_r.size()) * rot_floats * sizeof(float);
					for (uint32_t k = 0; k < cbones_t.size(); ++k)
					{
						const float* base = reinterpret_cast<const float*>(td + size_t(k) * 3 * sizeof(float));
						const uint32_t rec[2] = { cbones_t[k], 1u };
						std::fwrite(rec, sizeof(uint32_t), 2, qf); std::fwrite(base, sizeof(float), 3, qf);
					}
				}

				// ---- clip ranges: rotations in groups of 4 (min xxxx yyyy zzzz, extent xxxx yyyy zzzz), translations AOS (min3, extent3) ----
				{
					std::vector<uint32_t> abones_r, abones_t;
					for (uint32_t i = 0; i < num_out; ++i) { if (rot_types[i] == 2) abones_r.push_back(i); if (trans_types[i] == 2) abones_t.push_back(i); }
					const uint32_t count = is_variable ? uint32_t(abones_r.size() + abones_t.size()) : 0u;
					std::fwrite(&count, sizeof(uint32_t), 1, qf);
					if (is_variable)
					{
						const uint8_t* rd = th.get_clip_range_data();
						const uint32_t rot_floats = (settings.rotation_format == rotation_format8::quatf_full) ? 4 : 3;
						size_t off = 0;
						for (uint32_t k = 0; k < abones_r.size(); ++k)
						{
							const uint32_t g = k / 4, j = k % 4;
							const uint32_t gsize = std::min<uint32_t>(4, uint32_t(abones_r.size()) - g * 4);
							const float* base = reinterpret_cast<const float*>(rd + size_t(g) * 4 * rot_floats * 2 * sizeof(float));
							float mn[3], ex[3];
							for (uint32_t c = 0; c < 3; ++c) { mn[c] = base[c * gsize + j]; ex[c] = base[(rot_floats + c) * gsize + j]; }
							const uint32_t rec[2] = { abones_r[k], 0u };
							std::fwrite(rec, sizeof(uint32_t), 2, qf); std::fwrite(mn, sizeof(float), 3, qf); std::fwrite(ex, sizeof(float), 3, qf);
						}
						off = size_t(abones_r.size()) * rot_floats * 2 * sizeof(float);
						for (uint32_t k = 0; k < abones_t.size(); ++k)
						{
							const float* base = reinterpret_cast<const float*>(rd + off + size_t(k) * 6 * sizeof(float));
							const uint32_t rec[2] = { abones_t[k], 1u };
							std::fwrite(rec, sizeof(uint32_t), 2, qf); std::fwrite(base, sizeof(float), 6, qf);
						}
					}
				}
			}

			writer["segments"] = [&](sjson::ArrayWriter& segs)
			{
				for (uint32_t s = 0; s < num_segments; ++s)
				{
					const acl_impl::segment_header* sh;
					uint32_t sample_indices = 0xFFFFFFFFu;
					if (stripped)
					{
						const acl_impl::stripped_segment_header_t& ssh = th.get_stripped_segment_headers()[s];
						sh = &ssh;
						sample_indices = ssh.sample_indices;
					}
					else
						sh = &th.get_segment_headers()[s];

					const uint8_t* fmt; const uint8_t* range; const uint8_t* anim;
					th.get_segment_data(*sh, fmt, range, anim);

					const uint32_t first = seg_start ? seg_start[s] : 0;
					// The delimiting entry after the last segment is 0xFFFFFFFF, use the stored sample count instead
					const uint32_t stored_samples = hdr.num_samples;
					const uint32_t count = seg_start ? ((s + 1 < num_segments) ? (seg_start[s + 1] - seg_start[s]) : (stored_samples - seg_start[s])) : stored_samples;

					segs.push([&](sjson::ObjectWriter& w)
					{
						w["first_sample"] = first;
						w["num_samples"] = count;
						w["animated_pose_bits"] = sh->animated_pose_bit_size;
						w["animated_rotation_bits"] = sh->animated_rotation_bit_size;
						w["animated_translation_bits"] = sh->animated_translation_bit_size;
						w["sample_indices_bitset"] = sample_indices;
						w["animated_data_offset"] = uint32_t(anim - reinterpret_cast<const uint8_t*>(&th));

						// Rotation format bytes: groups of 4, last group padded
						uint32_t idx = 0;
						std::vector<int32_t> rot_bits(num_out, -1), trans_bits(num_out, -1);
						if (is_rotation_format_variable(settings.rotation_format))
						{
							for (uint32_t i = 0; i < num_out; ++i)
								if (rot_types[i] == 2) { const uint8_t b = fmt[idx++]; rot_bits[i] = (b == 31) ? 32 : b; }
							idx = (idx + 3) / 4 * 4;
						}
						if (is_vector_format_variable(settings.translation_format))
						{
							for (uint32_t i = 0; i < num_out; ++i)
								if (trans_types[i] == 2) { const uint8_t b = fmt[idx++]; trans_bits[i] = (b == 31) ? 32 : b; }
						}
						w["rotation_bits"] = [&](sjson::ArrayWriter& a) { for (uint32_t i = 0; i < num_out; ++i) a.push(rot_bits[i]); };
						w["translation_bits"] = [&](sjson::ArrayWriter& a) { for (uint32_t i = 0; i < num_out; ++i) a.push(trans_bits[i]); };

						if (qf != nullptr)
						{
							// ---- segment record ----
							// entries: all animated rotation sub-tracks (output bone order) then translations; each: bone, nbits, r[6]
							//   nbits > 0 && has segment ranges: r = [min x,y,z, extent x,y,z] as 8-bit integers (value/255)
							//   nbits == 0 (constant in segment): r = [x16, y16, z16, 0,0,0]  (clip-normalized value * 65535)
							//   no segment ranges (single segment): r = zeros
							// then stored_samples x (entries with nbits > 0) x 3 uint32 symbols (nbits == 32: raw float32 bit patterns)
							const bool has_seg_ranges = num_segments > 1;
							std::vector<uint32_t> ent;	// bone, nbits, r[6]
							uint32_t nR = 0, nT = 0;
							std::vector<uint32_t> nbits_stream;
							for (uint32_t i = 0; i < num_out; ++i)
							{
								if (rot_types[i] != 2) continue;
								const uint32_t g = nR / 4, j = nR % 4;
								uint32_t r6[6] = { 0, 0, 0, 0, 0, 0 };
								if (has_seg_ranges)
								{
									const uint8_t* gb = range + size_t(g) * 24;
									if (rot_bits[i] == 0)
									{
										r6[0] = (uint32_t(gb[j + 0]) << 8) | gb[j + 4];
										r6[1] = (uint32_t(gb[j + 8]) << 8) | gb[j + 12];
										r6[2] = (uint32_t(gb[j + 16]) << 8) | gb[j + 20];
									}
									else
										for (uint32_t c = 0; c < 6; ++c) r6[c] = gb[j + 4 * c];
								}
								ent.push_back(i); ent.push_back(uint32_t(rot_bits[i])); for (uint32_t c = 0; c < 6; ++c) ent.push_back(r6[c]);
								if (rot_bits[i] > 0) nbits_stream.push_back(uint32_t(rot_bits[i]));
								nR++;
							}
							const uint32_t num_rot_groups = (nR + 3) / 4;
							const uint8_t* trange = range + size_t(num_rot_groups) * 24;
							for (uint32_t i = 0; i < num_out; ++i)
							{
								if (trans_types[i] != 2) continue;
								uint32_t r6[6] = { 0, 0, 0, 0, 0, 0 };
								if (has_seg_ranges)
								{
									const uint8_t* b = trange + size_t(nT) * 6;
									if (trans_bits[i] == 0)
									{
										r6[0] = uint32_t(b[0]) | (uint32_t(b[1]) << 8);
										r6[1] = uint32_t(b[2]) | (uint32_t(b[3]) << 8);
										r6[2] = uint32_t(b[4]) | (uint32_t(b[5]) << 8);
									}
									else
										for (uint32_t c = 0; c < 6; ++c) r6[c] = b[c];
								}
								ent.push_back(i); ent.push_back(uint32_t(trans_bits[i])); for (uint32_t c = 0; c < 6; ++c) ent.push_back(r6[c]);
								if (trans_bits[i] > 0) nbits_stream.push_back(uint32_t(trans_bits[i]));
								nT++;
							}

							uint32_t stored_samples = count;
							if (stripped)
							{
								stored_samples = 0;
								for (uint32_t i = 0; i < count; ++i) if ((sample_indices >> (31 - i)) & 1) stored_samples++;
							}
							const uint32_t seg_hdr[6] = { first, count, stored_samples, sample_indices, nR, nT };
							std::fwrite(seg_hdr, sizeof(uint32_t), 6, qf);
							std::fwrite(ent.data(), sizeof(uint32_t), ent.size(), qf);

							uint64_t bit_offset = 0;
							auto read_bits = [&](uint32_t n) -> uint32_t
							{
								uint32_t value = 0;
								for (uint32_t k = 0; k < n; ++k)
								{
									const uint64_t b = bit_offset + k;
									value = (value << 1) | ((anim[b >> 3] >> (7 - (b & 7))) & 1);
								}
								bit_offset += n;
								return value;
							};
							const uint32_t num_stored = uint32_t(nbits_stream.size());
							std::vector<uint32_t> vals(size_t(num_stored) * 3);
							for (uint32_t smp = 0; smp < stored_samples; ++smp)
							{
								for (uint32_t k = 0; k < num_stored; ++k)
								{
									const uint32_t n = nbits_stream[k];
									vals[k * 3 + 0] = read_bits(n);
									vals[k * 3 + 1] = read_bits(n);
									vals[k * 3 + 2] = read_bits(n);
								}
								if (num_stored) std::fwrite(vals.data(), sizeof(uint32_t), vals.size(), qf);
							}
							if (stored_samples != 0 && bit_offset / stored_samples != sh->animated_pose_bit_size)
								std::printf("warning: segment %u pose bits mismatch %llu vs %u\n", s, (unsigned long long)(bit_offset / stored_samples), sh->animated_pose_bit_size);
						}
					});
				}
			};

			if (qf) std::fclose(qf);
		}
	}

	allocator.deallocate(compressed, compressed->get_size());
	std::fclose(out);
	return 0;
}
