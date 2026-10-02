/* Bit-exact C kernels of the shared numerics (curvecodec/fast/kernels.py builds and loads this file with enckern.c).
 *
 * Every function mirrors a numpy function of the package expression for expression, in the same float64 operation
 * order, so the results are identical to the last bit (compiled with -ffp-contract=off and without -ffast-math: no
 * FMA contraction, no reassociation). Integer functions mirror numpy's int64 semantics (wrap-around arithmetic,
 * arithmetic right shift, floor division). The Python side checks dtypes / contiguity and falls back to numpy for
 * anything unusual, so these functions may assume well-formed inputs.
 *
 *   np_sum              numpy's float64 add.reduce (pairwise summation)
 *   nlms_run            predict.nlms_residuals (integer NLMS predictor)
 *   model_forward       rans.model_forward (the adaptive context model: (start, freq) per symbol)
 *   features_batch,     features.CurveState: the 36 Q8 integer features of the coded residuals (whole curves / the
 *   window_features     training windows)
 *   interp_vectors      keys.interp_vectors (Catmull-Rom / linear interpolation of the key values)
 *   symbol_freqs        features.symbol_freqs (the mixture CDF -> the frequency tables of the alphabet)
 */
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

typedef int64_t i64;
typedef uint64_t u64;

/* ------------------------------------------------------------------------------------------------------------ */
/* numpy float64 pairwise summation (numpy/core/src/umath/loops_utils.h.src, PW_BLOCKSIZE 128)                   */
/* ------------------------------------------------------------------------------------------------------------ */
static double pw_sum(const double *a, i64 n, i64 stride)
{
    if (n < 8) {
        double res = 0.;
        for (i64 i = 0; i < n; i++) res += a[i * stride];
        return res;
    } else if (n <= 128) {
        double r[8], res;
        i64 i;
        for (int j = 0; j < 8; j++) r[j] = a[j * stride];
        for (i = 8; i < n - (n % 8); i += 8) {
            r[0] += a[(i + 0) * stride]; r[1] += a[(i + 1) * stride];
            r[2] += a[(i + 2) * stride]; r[3] += a[(i + 3) * stride];
            r[4] += a[(i + 4) * stride]; r[5] += a[(i + 5) * stride];
            r[6] += a[(i + 6) * stride]; r[7] += a[(i + 7) * stride];
        }
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i * stride];
        return res;
    } else {
        i64 n2 = n / 2;
        n2 -= n2 % 8;
        return pw_sum(a, n2, stride) + pw_sum(a + n2 * stride, n - n2, stride);
    }
}

/* the reduction as numpy's add.reduce runs it on a contiguous 1-d float64 array (variant selected at load time by
 * the Python self-test: 0 = identity + pairwise(a[0:n]), 1 = a[0] + pairwise(a[1:n])) */
static int g_sum_variant = 0;

void set_sum_variant(int v) { g_sum_variant = v; }

double np_sum(const double *a, i64 n)
{
    if (n <= 0) return 0.0;
    if (g_sum_variant == 1) {
        double io = a[0];
        io += pw_sum(a + 1, n - 1, 1);
        return io;
    }
    double io = 0.0;
    io += pw_sum(a, n, 1);
    return io;
}

/* ------------------------------------------------------------------------------------------------------------ */
/* integer helpers with numpy int64 semantics                                                                     */
/* ------------------------------------------------------------------------------------------------------------ */
static inline i64 wadd(i64 a, i64 b) { return (i64)((u64)a + (u64)b); }
static inline i64 wsub(i64 a, i64 b) { return (i64)((u64)a - (u64)b); }
static inline i64 wmul(i64 a, i64 b) { return (i64)((u64)a * (u64)b); }
static inline i64 shl(i64 a, int s) { return (i64)((u64)a << s); }
static inline i64 asr(i64 a, int s) { return a >> s; }          /* arithmetic on every compiler we build with */
static inline i64 floordiv(i64 a, i64 b)
{
    /* numpy floor_divide for int64 (b != 0; b == -1 with a == INT64_MIN gives INT64_MIN as numpy does) */
    if (b == -1) return (i64)(0 - (u64)a);
    i64 q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1;
    return q;
}
static inline i64 clipi(i64 v, i64 lo, i64 hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* ------------------------------------------------------------------------------------------------------------ */
/* nlms_run: A [F, C] int64 (row-major), x0 [C]; out [F, C] (encode: residuals, out[0] = 0; decode: X)            */
/* ------------------------------------------------------------------------------------------------------------ */
void nlms_run(int F, int C, const i64 *A, const i64 *x0, int order, int mu_shift, int decode, int init_d2,
              int nlms_sh, i64 clamp, i64 wmax, i64 *out)
{
    const i64 half = (i64)1 << (nlms_sh - 1);
    const int sh = nlms_sh - mu_shift;
    i64 hist[64], w[64];
    if (order > 64) return;
    memset(out, 0, sizeof(i64) * (size_t)F * C);
    for (int c = 0; c < C; c++) {
        for (int k = 0; k < order; k++) { hist[k] = 0; w[k] = 0; }
        if (init_d2) w[0] = (i64)1 << nlms_sh;
        i64 xprev = x0[c];
        if (decode && F > 0) out[c] = xprev;
        for (int t = 1; t < F; t++) {
            i64 acc = 0;
            for (int k = 0; k < order; k++) acc = wadd(acc, wmul(w[k], hist[k]));
            i64 P = wadd(xprev, asr(wadd(acc, half), nlms_sh));
            i64 x;
            if (decode) {
                x = wadd(A[(size_t)t * C + c], P);
                out[(size_t)t * C + c] = x;
            } else {
                x = A[(size_t)t * C + c];
                out[(size_t)t * C + c] = wsub(x, P);
            }
            i64 e = clipi(wsub(x, P), -clamp, clamp);
            i64 d = wsub(x, xprev);
            i64 norm = 0;
            i64 hc[64];
            for (int k = 0; k < order; k++) { hc[k] = clipi(hist[k], -clamp, clamp); norm = wadd(norm, wmul(hc[k], hc[k])); }
            norm = wadd(norm, order);
            for (int k = 0; k < order; k++) {
                i64 u = floordiv(shl(wmul(e, hc[k]), sh), norm);
                w[k] = clipi(wadd(w[k], u), -wmax, wmax);
            }
            for (int k = order - 1; k > 0; k--) hist[k] = hist[k - 1];
            if (order > 0) hist[0] = d;
            xprev = x;
        }
    }
}

/* ------------------------------------------------------------------------------------------------------------ */
/* model_forward: the adaptive context model (counts + inc per symbol, halved above the limit)                */
/* ------------------------------------------------------------------------------------------------------------ */
void model_forward(const i64 *sym, const i64 *ctx, i64 n, int nsym, int nctx, i64 scale, i64 inc, i64 limit,
                   const i64 *init, i64 *starts, i64 *freqs, i64 *cnts, i64 *tots)
{
    for (int c = 0; c < nctx; c++) {
        i64 t = 0;
        for (int j = 0; j < nsym; j++) { i64 v = init ? init[(size_t)c * nsym + j] : 1; cnts[(size_t)c * nsym + j] = v; t += v; }
        tots[c] = t;
    }
    for (i64 i = 0; i < n; i++) {
        const i64 c = ctx[i], s = sym[i];
        i64 *cnt = cnts + (size_t)c * nsym;
        i64 tot = tots[c], qsum = 0, before = 0, qs = 0;
        for (int j = 0; j < nsym; j++) {
            i64 q = floordiv(cnt[j] * scale, tot);
            qsum += q;
            if (j < s) before += q;
            if (j == s) qs = q;
        }
        i64 rem = scale - qsum;
        if (s == 0) { starts[i] = 0; freqs[i] = qs + 1 + rem; }
        else { starts[i] = before + s + rem; freqs[i] = qs + 1; }
        cnt[s] += inc; tot += inc;
        if (tot > limit) {
            i64 t2 = 0;
            for (int j = 0; j < nsym; j++) { cnt[j] = (cnt[j] + 1) >> 1; t2 += cnt[j]; }
            tot = t2;
        }
        tots[c] = tot;
    }
}

/* ------------------------------------------------------------------------------------------------------------ */
/* features.CurveState -- the 36 Q8 integer features, one curve at a time                                        */
/* ------------------------------------------------------------------------------------------------------------ */
#define NF 36
#define HH 8
#define NTAU 5

static const i64 *g_log_tab = 0;        /* LOG_TAB [256] (shipped table, set by the loader) */

void set_log_tab(const i64 *tab) { g_log_tab = tab; }

static inline int bitlen(u64 v) { return v ? 64 - __builtin_clzll(v) : 0; }

/* features.ilog2_q8: v = max(v, 1); e = bitlen(v) - 1; f = (e >= 8 ? v >> (e - 8) : v << (8 - e)) - 256;
 * e * 256 + LOG_TAB[clip(f, 0, 255)] */
static inline i64 ilog2_q8(i64 v)
{
    if (v < 1) v = 1;
    int e = bitlen((u64)v) - 1;
    i64 f = (e >= 8 ? (v >> (e - 8)) : (i64)((u64)v << (8 - e))) - 256;
    if (f < 0) f = 0;
    if (f > 255) f = 255;
    return (i64)e * 256 + g_log_tab[f];
}
static inline i64 Lq(i64 v) { return ilog2_q8(v + 1); }
static inline i64 slog2q(i64 x) { return x > 0 ? Lq(x) : (x < 0 ? -Lq(-x) : 0); }

/* The CurveState recursion of one curve for t = 0 .. T-1 (T <= n): r residuals, x [T + 1] decoded integers, lg =
 * L(max(gap, 1) - 1) per residual (NULL: an all-keys curve), nmax = the longest curve of the clip's lockstep batch,
 * base [36] = the static features (depth one-hot, kind, kf, pid one-hot, step, fps, p, len). Rows t >= t_lo are written
 * to outi (int16, row t - t_lo) or outf (float32). Returns 1 if a feature left +-32000. */
static int feat_run(i64 n, i64 nmax, const i64 *r, const i64 *x, const i64 *lg, const i64 *base, i64 T, i64 t_lo,
                    float *outf, int16_t *outi)
{
    /* the shift registers hold the already-transformed values (slog2q of the last residuals / differences), so each
     * step computes one new slog2q per register instead of re-transforming the whole history (same numbers) */
    i64 lh[HH] = {0}, ew[NTAU], ld[3] = {0}, la[2] = {0}, d0 = 0;
    for (int k = 0; k < NTAU; k++) ew[k] = 256;
    i64 xprev = x[0];
    int bad = 0;
    for (i64 t = 0; t < T; t++) {
        if (t >= t_lo) {
            i64 F[NF];
            for (int k = 0; k < NF; k++) F[k] = base[k];
            for (int k = 0; k < HH; k++) F[k] = lh[k];
            for (int k = 0; k < NTAU; k++) F[HH + k] = ilog2_q8(256 + ew[k]) - 8 * 256;
            int c = HH + NTAU;
            F[c] = ld[0]; F[c + 1] = ld[1]; F[c + 2] = ld[2];
            F[c + 3] = la[0]; F[c + 4] = la[1];
            c += 5;
            if (lg) {
                /* CurveState indexes the lockstep batch's gap matrix [nmax, C] (zero-padded past this curve's end,
                 * L(0) = 0 there): gb = lg[min(t, nmax - 1)], ga = lg[min(t + 1, nmax - 1)] */
                i64 kb = t < nmax - 1 ? t : nmax - 1;
                i64 ka = (t + 1) < nmax - 1 ? (t + 1) : nmax - 1;
                i64 gb = kb < n ? lg[kb] : 0;
                i64 ga = ka < n ? lg[ka] : 0;
                i64 gp = t > 0 ? lg[t - 1] : gb;
                F[c] = gb; F[c + 1] = ga; F[c + 2] = gb - gp;
            } else {
                F[c] = 0; F[c + 1] = 0; F[c + 2] = 0;
            }
            F[c + 3 + 13] = Lq(t);
            const size_t row = (size_t)(t - t_lo) * NF;
            for (int k = 0; k < NF; k++) {
                i64 v = F[k];
                if (v > 32000 || v < -32000) bad = 1;
                if (outi) outi[row + k] = (int16_t)v;
                else outf[row + k] = (float)(int16_t)v;
            }
        }
        /* update(r_t, x_{t+1}) */
        i64 rt = r[t];
        i64 u = rt < 0 ? -rt : rt;
        for (int k = HH - 1; k > 0; k--) lh[k] = lh[k - 1];
        lh[0] = slog2q(rt);
        for (int k = 0; k < NTAU; k++) ew[k] += asr(shl(u, 8) - ew[k], k + 1);
        i64 xn = x[t + 1];
        i64 dn = xn - xprev;
        la[1] = la[0]; la[0] = slog2q(dn - d0);
        ld[2] = ld[1]; ld[1] = ld[0]; ld[0] = slog2q(dn);
        d0 = dn;
        xprev = xn;
    }
    return bad;
}

int curve_features(i64 n, i64 nmax, const i64 *r, const i64 *x, const i64 *lg, const i64 *base, int16_t *out)
{
    return feat_run(n, nmax, r, x, lg, base, n, 0, 0, out);
}

/* The training windows: the features of rows t_lo .. T-1 of one curve (T <= n) as float32 [T - t_lo, 36]; x = the
 * decoded integers [T + 1], lg = L(max(gap, 1) - 1) per residual [min(T + 1, n)].  Returns 1 if a feature left int16. */
int window_features(i64 n, i64 nmax, const i64 *r, const i64 *x, const i64 *lg, const i64 *base, i64 T, i64 t_lo, float *out)
{
    return feat_run(n, nmax, r, x, lg, base, T, t_lo, out, 0);
}

/* ------------------------------------------------------------------------------------------------------------ */
/* keys.interp_vectors: K int64 [nk] sorted (K[0] = 0, K[-1] = F - 1), vk float64 [nk, 3] -> out [F, 3]          */
/* cubic (Catmull-Rom tangents on the key times, Hermite basis) when cubic != 0 and nk >= 3, else linear.        */
/* ------------------------------------------------------------------------------------------------------------ */
void interp_vectors(i64 nk, const i64 *K, const double *vk, i64 F, int cubic, double *m, double *out)
{
    const int use_cubic = cubic && nk >= 3;
    if (use_cubic) {
        for (i64 k = 1; k < nk - 1; k++) {
            double dt = (double)K[k + 1] - (double)K[k - 1];
            for (int c = 0; c < 3; c++) m[k * 3 + c] = (vk[(k + 1) * 3 + c] - vk[(k - 1) * 3 + c]) / dt;
        }
        double d0 = (double)K[1] - (double)K[0], d1 = (double)K[nk - 1] - (double)K[nk - 2];
        for (int c = 0; c < 3; c++) {
            m[c] = (vk[3 + c] - vk[c]) / d0;
            m[(nk - 1) * 3 + c] = (vk[(nk - 1) * 3 + c] - vk[(nk - 2) * 3 + c]) / d1;
        }
    }
    i64 j = 0;                                   /* searchsorted(K, f, 'right') - 1, clipped to [0, nk - 2] */
    for (i64 f = 0; f < F; f++) {
        while (j + 1 < nk && K[j + 1] <= f) j++;
        i64 jj = j < 0 ? 0 : j;
        if (jj > nk - 2) jj = nk - 2;
        const double t = (double)(f - K[jj]) / (double)(K[jj + 1] - K[jj]);
        const double *a = vk + jj * 3, *b = vk + (jj + 1) * 3;
        double *o = out + f * 3;
        if (use_cubic) {
            const double h = (double)K[jj + 1] - (double)K[jj];
            const double t2 = t * t, t3 = t2 * t;
            const double c0 = 2 * t3 - 3 * t2 + 1, c1 = t3 - 2 * t2 + t, c2 = -2 * t3 + 3 * t2, c3 = t3 - t2;
            const double *mj = m + jj * 3, *mj1 = m + (jj + 1) * 3;
            for (int c = 0; c < 3; c++) o[c] = c0 * a[c] + c1 * h * mj[c] + c2 * b[c] + c3 * h * mj1[c];
        } else {
            for (int c = 0; c < 3; c++) o[c] = a[c] + t * (b[c] - a[c]);
        }
    }
    for (i64 k = 0; k < nk; k++)
        for (int c = 0; c < 3; c++) out[K[k] * 3 + c] = vk[k * 3 + c];
}

/* Features of the C curves of one clip (the lockstep batch): n [C], r [N] (curve-major), dx [N] (x_{t+1} - x_t),
 * x0 [C], lg [N] (L(max(gap, 1) - 1) per residual; zeros for all-keys curves), base [C, 36], nmax = max n.
 * out int16 [N, 36].  Returns the number of curves with a feature outside +-32000 (the caller raises). */
i64 features_batch(i64 C, const i64 *n, i64 nmax, const i64 *r, const i64 *dx, const i64 *x0, const i64 *lg,
                   const i64 *base, int16_t *out, i64 *xbuf)
{
    i64 off = 0, bad = 0;
    for (i64 c = 0; c < C; c++) {
        const i64 nc = n[c];
        xbuf[0] = x0[c];
        for (i64 t = 0; t < nc; t++) xbuf[t + 1] = xbuf[t] + dx[off + t];
        bad += curve_features(nc, nmax, r + off, xbuf, lg + off, base + c * NF, out + (size_t)off * NF);
        off += nc;
    }
    return bad;
}

/* ------------------------------------------------------------------------------------------------------------ */
/* features.symbol_freqs (mixture_weights_q16 + cdf_q16): the rANS frequency tables of the model -- all integer    */
/* arithmetic, exactly as numpy.                                                                                  */
/* ------------------------------------------------------------------------------------------------------------ */
int symbol_freqs(i64 N, i64 K, i64 S, const i64 *logw, const i64 *mu, const i64 *logs, const i64 *lo, const i64 *hi,
                 const i64 *exp_tab, i64 exp_lim, const i64 *invs_tab, i64 logs_lo, const i64 *sig_tab, i64 sig_lim, i64 mu_lim,
                 int rans_scale, i64 *out, i64 *work)
{
    /* work: K (w) + K (inv_s) + 2 S (F) int64 */
    i64 *w = work, *invs = work + K, *Fv = work + 2 * K;
    const i64 M = (i64)1 << rans_scale;
    if (K > 64) return -1;
    for (i64 r = 0; r < N; r++) {
        const i64 *lw = logw + r * K, *m = mu + r * K, *ls = logs + r * K;
        /* mixture_weights_q16 */
        i64 mx = lw[0];
        for (i64 k = 1; k < K; k++) if (lw[k] > mx) mx = lw[k];
        i64 e[64], tot = 0, am = 0;
        for (i64 k = 0; k < K; k++) {
            i64 z = lw[k] - mx; if (z < -exp_lim) z = -exp_lim; if (z > 0) z = 0;
            e[k] = exp_tab[z + exp_lim]; tot += e[k];
            if (e[k] > e[am]) am = k;                                   /* np.argmax: the first maximum */
        }
        i64 wsum = 0;
        for (i64 k = 0; k < K; k++) { w[k] = (e[k] << 16) / tot; wsum += w[k]; }
        w[am] += 65536 - wsum;
        for (i64 k = 0; k < K; k++) invs[k] = invs_tab[ls[k] - logs_lo];
        /* cdf_q16 at the 2 S points (hi, then lo - 1) */
        for (i64 q = 0; q < 2 * S; q++) {
            i64 pt = q < S ? hi[q] : lo[q - S] - 1;
            i64 d = (i64)((u64)pt << 8) + 128;
            if (d < -mu_lim) d = -mu_lim; if (d > mu_lim) d = mu_lim;
            i64 acc = 0;
            for (i64 k = 0; k < K; k++) {
                i64 z = d - m[k];
                if (z < -mu_lim) z = -mu_lim; if (z > mu_lim) z = mu_lim;
                z = (z * invs[k]) >> 16;
                if (z < -sig_lim) z = -sig_lim; if (z > sig_lim) z = sig_lim;
                acc += sig_tab[z + sig_lim] * w[k];
            }
            Fv[q] = acc >> 16;
        }
        i64 ptot = 0;
        i64 *f = out + r * S;
        for (i64 s = 0; s < S; s++) { i64 P = Fv[s] - Fv[S + s]; if (P < 0) P = 0; f[s] = P; ptot += P; }
        const i64 den = ptot > 1 ? ptot : 1;
        i64 fsum = 0;
        for (i64 s = 0; s < S; s++) { f[s] = 1 + (f[s] * (M - S)) / den; fsum += f[s]; }
        f[0] += M - fsum;
    }
    return 0;
}
