/* Bit-exact C kernels of the encoder's search, linked with fastkern.c into one library (curvecodec/fast/kernels.py
 * builds this file with -fno-math-errno on top of the common flags: sqrt then compiles to the IEEE instruction without
 * the errno branch, which lets the element-wise loops vectorise; no result changes -- sqrt, +, -, *, / are correctly
 * rounded at any vector width and FMA contraction stays off).
 *
 * Every function mirrors the numpy code of the package expression for expression (same float64 operation order); the
 * Python side checks dtypes / layouts and falls back to numpy otherwise.
 *
 *   ek_subtree_eval     search.Search.eval_subtree / search.eval_subtree_frames: object-space FK + ACL's shell error
 *                       over a bone's subtree (structure-of-arrays, one loop per stage, the children read their parent's
 *                       rows from the call's own buffers)
 *   ek_trig / ek_sincos numpy's own float64 sin / cos inner loops (pointers set and self-tested by the loader):
 *                       exp_map in C gives numpy's bits
 *   ek_rk_rd            keys.remove_keys (the RD key removal of one track at one lambda)
 *   ek_ladder_c         keys.ladder_c (the per-component ladder)
 *   ek_reconstruct      keys.reconstruct
 *   ek_sess_*           repeated evaluations of one track's candidate steps (keys.grow_step)
 *   ek_commit_rows      search.Search.commit_pending
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int64_t i64;


extern double np_sum(const double *a, i64 n);          /* fastkern.c: numpy's pairwise add.reduce */

/* growable scratch buffers that live for the process (the encoder is single-threaded; they only grow) */
typedef struct { void *p; size_t n; } ek_buf;
static void *bget(ek_buf *b, size_t bytes)
{
    if (bytes > b->n) {
        free(b->p);
        b->n = bytes + bytes / 2 + 64;
        b->p = malloc(b->n);
        if (!b->p) b->n = 0;
    }
    return b->p;
}

/* ------------------------------------------------------------------------------------------------------------ */
/* numpy's float64 sin / cos (PyUFuncGenericFunction: void f(char **args, npy_intp const *dims, npy_intp const   */
/* *steps, void *data)), called on contiguous arrays                                                             */
/* ------------------------------------------------------------------------------------------------------------ */
typedef void (*ufunc_loop)(char **args, const i64 *dims, const i64 *steps, void *data);
static ufunc_loop g_sin = 0, g_cos = 0;
static void *g_sin_data = 0, *g_cos_data = 0;

void ek_trig(void *sin_loop, void *sin_data, void *cos_loop, void *cos_data)
{
    g_sin = (ufunc_loop)sin_loop; g_sin_data = sin_data;
    g_cos = (ufunc_loop)cos_loop; g_cos_data = cos_data;
}

int ek_trig_ok(void) { return g_sin != 0 && g_cos != 0; }

/* s = sin(x), c = cos(x) (either output may be NULL), n contiguous doubles, exactly numpy's np.sin / np.cos */
void ek_sincos(i64 n, const double *x, double *s, double *c)
{
    if (n <= 0) return;
    i64 dims[1] = {n}, steps[2] = {8, 8};
    if (s) { char *args[2] = {(char *)x, (char *)s}; g_sin(args, dims, steps, g_sin_data); }
    if (c) { char *args[2] = {(char *)x, (char *)c}; g_cos(args, dims, steps, g_cos_data); }
}

/* every element finite (no inf / NaN), by the exponent bits: branch-free, so the compiler vectorises it */
static int allfin_d(const double *x, size_t n)
{
    uint64_t bad = 0;
    for (size_t i = 0; i < n; i++) { uint64_t u; memcpy(&u, x + i, 8); bad |= (u & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL; }
    return bad == 0;
}

static int allfin_f(const float *x, size_t n)
{
    uint32_t bad = 0;
    for (size_t i = 0; i < n; i++) { uint32_t u; memcpy(&u, x + i, 4); bad |= (u & 0x7f800000U) == 0x7f800000U; }
    return bad == 0;
}

/* per-bone error statistics: *mx = the max (the first element, then every larger one -- numpy's max when there is no
 * NaN), *ct = the number of elements > p.  Returns 1 when a NaN is present (outputs unset).  Branch-free: the max of
 * non-negative doubles is the max of their bit patterns (the shell errors are sqrt of sums of squares: +0 or more);
 * a negative element (never produced by the kernel) takes the scalar loop. */
static int err_stats(const double *e, int n, double p, double *mx, double *ct)
{
    uint64_t nanb = 0, neg = 0, umx = 0;
    int64_t c = 0;
    for (int t = 0; t < n; t++) {
        uint64_t u; memcpy(&u, e + t, 8);
        nanb |= (u & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL;
        neg |= u >> 63;
        umx = u > umx ? u : umx;
        c += e[t] > p;
    }
    if (nanb) return 1;
    if (neg) {
        double m = 0.0;
        for (int t = 0; t < n; t++) if (t == 0 || e[t] > m) m = e[t];
        *mx = m;
    } else {
        memcpy(mx, &umx, 8);
    }
    *ct = (double)c;
    return 0;
}

/* ------------------------------------------------------------------------------------------------------------ */
/* the vectorisable stages of the subtree kernel (the expressions of fastkern.c::subtree_eval, split per stage)  */
/* ------------------------------------------------------------------------------------------------------------ */
/* rot.quat_normalize of the local rotation (root bone: object rotation = normalised local, position = local t) */
static void st_norm(int n, double *restrict q0, double *restrict q1, double *restrict q2, double *restrict q3)
{
    for (int t = 0; t < n; t++) {
        double s = 0.0;
        s += q0[t] * q0[t]; s += q1[t] * q1[t]; s += q2[t] * q2[t]; s += q3[t] * q3[t];
        double nn = sqrt(s);
        nn = nn < 1e-30 ? 1e-30 : nn;                  /* np.maximum(n, 1e-30), NaN propagates */
        q0[t] = q0[t] / nn; q1[t] = q1[t] / nn; q2[t] = q2[t] / nn; q3[t] = q3[t] / nn;
    }
}

/* rot = normalize(quat_mul(normalize(lq), parent_rot)), pos = rotate(lt, parent_rot) + parent_pos (in place in q / t) */
static void st_fk(int n, double *restrict q0, double *restrict q1, double *restrict q2, double *restrict q3,
                  double *restrict t0, double *restrict t1, double *restrict t2,
                  const double *restrict p0, const double *restrict p1, const double *restrict p2, const double *restrict p3,
                  const double *restrict s0, const double *restrict s1, const double *restrict s2)
{
    for (int t = 0; t < n; t++) {
        double s = 0.0;
        s += q0[t] * q0[t]; s += q1[t] * q1[t]; s += q2[t] * q2[t]; s += q3[t] * q3[t];
        double nn = sqrt(s);
        nn = nn < 1e-30 ? 1e-30 : nn;
        const double lx = q0[t] / nn, ly = q1[t] / nn, lz = q2[t] / nn, lw = q3[t] / nn;
        const double bx = p0[t], by = p1[t], bz = p2[t], bw = p3[t];
        const double qx = bw * lx + bx * lw + by * lz - bz * ly;
        const double qy = bw * ly - bx * lz + by * lw + bz * lx;
        const double qz = bw * lz + bx * ly - by * lx + bz * lw;
        const double qw = bw * lw - bx * lx - by * ly - bz * lz;
        double s2_ = 0.0;
        s2_ += qx * qx; s2_ += qy * qy; s2_ += qz * qz; s2_ += qw * qw;
        double n2 = sqrt(s2_);
        n2 = n2 < 1e-30 ? 1e-30 : n2;
        const double vx = t0[t], vy = t1[t], vz = t2[t];
        const double tx = 2.0 * (by * vz - bz * vy);
        const double ty = 2.0 * (bz * vx - bx * vz);
        const double tz = 2.0 * (bx * vy - by * vx);
        t0[t] = vx + bw * tx + (by * tz - bz * ty) + s0[t];
        t1[t] = vy + bw * ty + (bz * tx - bx * tz) + s1[t];
        t2[t] = vz + bw * tz + (bx * ty - by * tx) + s2[t];
        q0[t] = qx / n2; q1[t] = qy / n2; q2[t] = qz / n2; q3[t] = qw / n2;
    }
}

/* ACL's shell error: sqrt(max_axis sum_xyz (shell * (raw_col_axis - col_axis(rot)) + (raw_pos - pos))^2) */
static void st_err(int n, const double *restrict q0, const double *restrict q1, const double *restrict q2, const double *restrict q3,
                   const double *restrict t0, const double *restrict t1, const double *restrict t2,
                   const double *restrict c0, const double *restrict c1, const double *restrict c2,
                   const double *restrict c3, const double *restrict c4, const double *restrict c5,
                   const double *restrict c6, const double *restrict c7, const double *restrict c8,
                   const double *restrict r0, const double *restrict r1, const double *restrict r2,
                   double shell, double *restrict er)
{
    for (int t = 0; t < n; t++) {
        const double rx = q0[t], ry = q1[t], rz = q2[t], rw = q3[t], px = t0[t], py = t1[t], pz = t2[t];
        const double xx = rx * rx, yy = ry * ry, zz = rz * rz;
        const double xy = rx * ry, xz = rx * rz, yz = ry * rz, wx = rw * rx, wy = rw * ry, wz = rw * rz;
        const double m0 = 1 - 2 * (yy + zz), m1 = 2 * (xy + wz), m2 = 2 * (xz - wy);
        const double m3 = 2 * (xy - wz), m4 = 1 - 2 * (xx + zz), m5 = 2 * (yz + wx);
        const double m6 = 2 * (xz + wy), m7 = 2 * (yz - wx), m8 = 1 - 2 * (xx + yy);
        const double dx0 = r0[t] - px, dy0 = r1[t] - py, dz0 = r2[t] - pz;
        double ax, ay, az, sa, best;
        ax = shell * (c0[t] - m0) + dx0; ay = shell * (c1[t] - m1) + dy0; az = shell * (c2[t] - m2) + dz0;
        sa = 0.0; sa += ax * ax; sa += ay * ay; sa += az * az; best = sa;
        ax = shell * (c3[t] - m3) + dx0; ay = shell * (c4[t] - m4) + dy0; az = shell * (c5[t] - m5) + dz0;
        sa = 0.0; sa += ax * ax; sa += ay * ay; sa += az * az; best = sa > best ? sa : best;
        ax = shell * (c6[t] - m6) + dx0; ay = shell * (c7[t] - m7) + dy0; az = shell * (c8[t] - m8) + dz0;
        sa = 0.0; sa += ax * ax; sa += ay * ay; sa += az * az; best = sa > best ? sa : best;
        er[t] = sqrt(best);
    }
}

/* ------------------------------------------------------------------------------------------------------------ */
/* ek_subtree_eval: fastkern.c::subtree_eval with the same arguments and results, plus the constant raw arrays in   */
/* structure-of-arrays bone-major form (raw_cols_s [B, 9, F], raw_pos_s [B, 3, F]: a full-clip call reads them in    */
/* place) and aos_mode: 0 = the AoS outputs o_rot / o_pos of every computed bone are written (as subtree_eval);     */
/* 1 = only when every subtree bone was computed (an early-exit failure's partial outputs are never read by the     */
/* search -- try_local drops them).  The committed locals of the subtree are gathered frame by frame for 8 bones at */
/* a time (one frame's records of consecutive bones are contiguous in the frame-major state), the children read     */
/* their parent's rows from the call's own structure-of-arrays rows.  Scratch memory lives for the process.         */
/* ------------------------------------------------------------------------------------------------------------ */
static ek_buf g_k[4];

int ek_subtree_eval(int F, int B, int n, const i64 *idx,
                    int S, const int *sub, const int *par_rel, const int *par_abs, int parent_raw,
                    const float *new_local, const float *cur_local,
                    const double *obj_rot, const double *obj_pos,
                    const double *raw_rot, const double *raw_pos,
                    const double *raw_cols_s, const double *raw_pos_s, double shell,
                    int mode, const double *lim_max, const double *lim_sum, double p,
                    double *o_rot, double *o_pos, double *o_err,
                    double *o_max, double *o_sum, double *o_cnt, int aos_mode, int fexit)
{
    const size_t N = (size_t)n;
    int finite = fexit != 0;                             /* the frame-level exit needs finite inputs (checked on the way) */
    /* per bone of the call: 7 SoA rows (rot x y z w, pos x y z) kept for the children; + 7 parent rows + 12 raw rows */
    double *buf = (double *)bget(&g_k[0], sizeof(double) * N * (7 * (size_t)S + 19));
    if (!buf) return -2;
    double *pr_ = buf + 7 * N * (size_t)S;               /* parent rows of bone 0 (7 N) */
    double *rw = pr_ + 7 * N;                            /* gathered raw rows (12 N) */
    int ret = S;
    int gathered = 1;                                    /* bones [1, gathered) have their local rows in place */
    for (int j = 0; j < S; j++) {
        const int c = sub[j], pa = par_abs[j], prel = par_rel[j];
        double *R = buf + 7 * N * (size_t)j;
        double *q0 = R, *q1 = R + N, *q2 = R + 2 * N, *q3 = R + 3 * N, *t0 = R + 4 * N, *t1 = R + 5 * N, *t2 = R + 6 * N;
        /* local transform (float32 -> float64) */
        if (j == 0) {
            for (int t = 0; t < n; t++) {
                const i64 f = idx ? idx[t] : (i64)t;
                const float *lf = new_local + (size_t)f * 7;
                q0[t] = (double)lf[0]; q1[t] = (double)lf[1]; q2[t] = (double)lf[2]; q3[t] = (double)lf[3];
                t0[t] = (double)lf[4]; t1[t] = (double)lf[5]; t2[t] = (double)lf[6];
            }
            if (finite && !allfin_d(R, 7 * N)) finite = 0;
        } else if (j >= gathered) {
            /* the next (up to) 8 bones, frame by frame: their records of one frame are close together in memory */
            const int j1 = j + 8 < S ? j + 8 : S;
            for (int t = 0; t < n; t++) {
                const i64 f = idx ? idx[t] : (i64)t;
                const float *row = cur_local + (size_t)f * B * 7;
                for (int jj = j; jj < j1; jj++) {
                    const float *lf = row + (size_t)sub[jj] * 7;
                    double *RR = buf + 7 * N * (size_t)jj;
                    RR[t] = (double)lf[0]; RR[N + t] = (double)lf[1]; RR[2 * N + t] = (double)lf[2]; RR[3 * N + t] = (double)lf[3];
                    RR[4 * N + t] = (double)lf[4]; RR[5 * N + t] = (double)lf[5]; RR[6 * N + t] = (double)lf[6];
                }
            }
            gathered = j1;
            if (finite && !allfin_d(buf + 7 * N * (size_t)j, 7 * N * (size_t)(j1 - j))) finite = 0;
        }
        /* raw rows (constant, SoA bone-major): in place for a full-clip call, gathered for a frame list */
        const double *rc[9], *rp[3];
        if (idx) {
            for (int k = 0; k < 9; k++) {
                const double *src = raw_cols_s + ((size_t)c * 9 + k) * F;
                double *dst = rw + (size_t)k * N;
                for (int t = 0; t < n; t++) dst[t] = src[idx[t]];
                rc[k] = dst;
            }
            for (int k = 0; k < 3; k++) {
                const double *src = raw_pos_s + ((size_t)c * 3 + k) * F;
                double *dst = rw + (size_t)(9 + k) * N;
                for (int t = 0; t < n; t++) dst[t] = src[idx[t]];
                rp[k] = dst;
            }
        } else {
            for (int k = 0; k < 9; k++) rc[k] = raw_cols_s + ((size_t)c * 9 + k) * F;
            for (int k = 0; k < 3; k++) rp[k] = raw_pos_s + ((size_t)c * 3 + k) * F;
        }
        if (pa < 0) {
            st_norm(n, q0, q1, q2, q3);
        } else {
            const double *P0, *P1, *P2, *P3, *Q0, *Q1, *Q2;
            if (j > 0 && prel >= 0) {                     /* the parent was computed in this call */
                const double *PR = buf + 7 * N * (size_t)prel;
                P0 = PR; P1 = PR + N; P2 = PR + 2 * N; P3 = PR + 3 * N; Q0 = PR + 4 * N; Q1 = PR + 5 * N; Q2 = PR + 6 * N;
            } else {                                      /* the committed state (or the raw chain) */
                double *a0 = pr_, *a1 = pr_ + N, *a2 = pr_ + 2 * N, *a3 = pr_ + 3 * N, *b0 = pr_ + 4 * N, *b1 = pr_ + 5 * N, *b2 = pr_ + 6 * N;
                const int use_raw = (j == 0 && parent_raw);
                for (int t = 0; t < n; t++) {
                    const i64 f = idx ? idx[t] : (i64)t;
                    const double *prot = use_raw ? raw_rot + ((size_t)pa * F + f) * 4 : obj_rot + ((size_t)f * B + pa) * 4;
                    const double *ppos = use_raw ? raw_pos + ((size_t)pa * F + f) * 3 : obj_pos + ((size_t)f * B + pa) * 3;
                    a0[t] = prot[0]; a1[t] = prot[1]; a2[t] = prot[2]; a3[t] = prot[3];
                    b0[t] = ppos[0]; b1[t] = ppos[1]; b2[t] = ppos[2];
                }
                P0 = a0; P1 = a1; P2 = a2; P3 = a3; Q0 = b0; Q1 = b1; Q2 = b2;
                if (finite && !allfin_d(pr_, 7 * N)) finite = 0;
            }
            if (!(finite && mode >= 1)) st_fk(n, q0, q1, q2, q3, t0, t1, t2, P0, P1, P2, P3, Q0, Q1, Q2);
            else {
                /* frame blocks with the max test after each: a bone over its max limit in any block fails whatever
                 * the other frames hold (max is exact; every input is finite, so no NaN can hide in them) */
                double *ej = o_err + (size_t)j * N;
                int failed = 0;
                double bmx = 0.0;
                for (int t0_ = 0; t0_ < n && !failed; t0_ += 256) {
                    const int m = n - t0_ < 256 ? n - t0_ : 256;
                    st_fk(m, q0 + t0_, q1 + t0_, q2 + t0_, q3 + t0_, t0 + t0_, t1 + t0_, t2 + t0_, P0 + t0_, P1 + t0_, P2 + t0_, P3 + t0_,
                          Q0 + t0_, Q1 + t0_, Q2 + t0_);
                    st_err(m, q0 + t0_, q1 + t0_, q2 + t0_, q3 + t0_, t0 + t0_, t1 + t0_, t2 + t0_, rc[0] + t0_, rc[1] + t0_, rc[2] + t0_,
                           rc[3] + t0_, rc[4] + t0_, rc[5] + t0_, rc[6] + t0_, rc[7] + t0_, rc[8] + t0_, rp[0] + t0_, rp[1] + t0_, rp[2] + t0_,
                           shell, ej + t0_);
                    for (int t = t0_; t < t0_ + m; t++) if (ej[t] > bmx || t == 0) bmx = ej[t];
                    if (bmx > lim_max[j]) failed = 1;
                }
                if (failed) { o_max[j] = bmx; ret = j + 1; break; }
                goto stats;
            }
        }
        if (pa < 0 && finite && mode >= 1) {
            double *ej = o_err + (size_t)j * N;
            int failed = 0;
            double bmx = 0.0;
            for (int t0_ = 0; t0_ < n && !failed; t0_ += 256) {
                const int m = n - t0_ < 256 ? n - t0_ : 256;
                st_err(m, q0 + t0_, q1 + t0_, q2 + t0_, q3 + t0_, t0 + t0_, t1 + t0_, t2 + t0_, rc[0] + t0_, rc[1] + t0_, rc[2] + t0_,
                       rc[3] + t0_, rc[4] + t0_, rc[5] + t0_, rc[6] + t0_, rc[7] + t0_, rc[8] + t0_, rp[0] + t0_, rp[1] + t0_, rp[2] + t0_,
                       shell, ej + t0_);
                for (int t = t0_; t < t0_ + m; t++) if (ej[t] > bmx || t == 0) bmx = ej[t];
                if (bmx > lim_max[j]) failed = 1;
            }
            if (failed) { o_max[j] = bmx; ret = j + 1; break; }
            goto stats;
        }
        {
        double *ej = o_err + (size_t)j * N;
        st_err(n, q0, q1, q2, q3, t0, t1, t2, rc[0], rc[1], rc[2], rc[3], rc[4], rc[5], rc[6], rc[7], rc[8], rp[0], rp[1], rp[2], shell, ej);
        }
stats:;
        double *ej = o_err + (size_t)j * N;
        double mx = 0.0, ct = 0.0;
        if (err_stats(ej, n, p, &mx, &ct)) { ret = -1; break; }
        o_max[j] = mx; o_cnt[j] = ct;
        o_sum[j] = np_sum(ej, n);
        if (mode == 1 && n > 0 && mx > lim_max[j]) { ret = j + 1; break; }
        if (mode == 2 && n > 0 && (o_sum[j] > lim_sum[j] || mx > lim_max[j])) { ret = j + 1; break; }
        /* mode 3 (the ladder's step-growth offer, keys.grow_step): the offer is dropped as soon as
         * a bone is over its guard or its added error D_j = sum_j - esum_j does not fit its cap: lim_sum = [esum_j,
         * rem_sub_j] pairs, the test exactly numpy's `D <= rem_sub + 1e-12` */
        if (mode == 3 && n > 0 && (mx > lim_max[j] || !(o_sum[j] - lim_sum[2 * j] <= lim_sum[2 * j + 1] + 1e-12))) { ret = j + 1; break; }
    }
    /* the AoS outputs */
    if (ret >= 0 && (aos_mode == 0 || ret == S)) {
        for (int j = 0; j < ret; j++) {
            const double *R = buf + 7 * N * (size_t)j;
            double *rj = o_rot + (size_t)j * N * 4, *pj = o_pos + (size_t)j * N * 3;
            for (size_t t = 0; t < N; t++) {
                rj[4 * t] = R[t]; rj[4 * t + 1] = R[N + t]; rj[4 * t + 2] = R[2 * N + t]; rj[4 * t + 3] = R[3 * N + t];
                pj[3 * t] = R[4 * N + t]; pj[3 * t + 1] = R[5 * N + t]; pj[3 * t + 2] = R[6 * N + t];
            }
        }
    }
    return ret;
}

/* ============================================================================================================== */
/* The key-removal loops. The search state is the numpy state itself (the frame-major arrays of search.Search,     */
/* read and written in place, exactly the rows the numpy code writes); the constant raw arrays are the bone-major  */
/* structure-of-arrays copies of kernels._cache.                                                                    */
/* ============================================================================================================== */
typedef struct {
    int F, B;
    float *cur_local;                   /* [F, B, 7] */
    double *obj_rot, *obj_pos, *err;    /* [F, B, 4] [F, B, 3] [F, B] */
    double *emax, *cnt, *esum;          /* [B] */
    const double *limits, *mlim, *n_allow;   /* [B]; n_allow unused */
    const double *raw_cols_s, *raw_pos_s;    /* [B, 9, F] [B, 3, F] */
    double shell, p;
    const double *raw_a;                     /* [B, F, 12]: raw cols (9) + raw pos (3) per frame (windowed evaluations) */
} ek_state;

typedef struct {
    int b, kind, nsub;                  /* kind 0 = rotation (log-map, exp_map), 1 = translation */
    const int *sub, *par_rel, *par_abs; /* the subtree plan (fast._subtree_plan) */
    const double *v;                    /* [F, 3] the raw track values (vals[tr]) */
} ek_track;

static ufunc_loop g_log2 = 0;
static void *g_log2_data = 0;

void ek_log2set(void *loop, void *data) { g_log2 = (ufunc_loop)loop; g_log2_data = data; }

void ek_log2(i64 n, const double *x, double *y)
{
    if (n <= 0) return;
    i64 dims[1] = {n}, steps[2] = {8, 8};
    char *args[2] = {(char *)x, (char *)y};
    g_log2(args, dims, steps, g_log2_data);
}


/* numpy quantize / dequantize of one value: rint(v / step) -> int64 ; float32(n * step) -> float64 */
static inline i64 q_int(double v, double s) { return (i64)nearbyint(v / s); }
static inline double dq(i64 n, double s) { return (double)(float)((double)n * s); }

/* keys.interp_vectors of ONE column at the frames rows[0..n) (increasing): K [nk] keys, vk [nk] values
 * (float64), m [nk] tangent scratch.  The same expressions per frame as interp_vectors / fastkern.c::interp_rows. */
static void interp_col(i64 nk, const i64 *K, const double *vk, i64 n, const i64 *rows, int cubic, double *m, double *out)
{
    const int use_cubic = cubic && nk >= 3;
    if (use_cubic) {
        for (i64 k = 1; k < nk - 1; k++) m[k] = (vk[k + 1] - vk[k - 1]) / ((double)K[k + 1] - (double)K[k - 1]);
        m[0] = (vk[1] - vk[0]) / ((double)K[1] - (double)K[0]);
        m[nk - 1] = (vk[nk - 1] - vk[nk - 2]) / ((double)K[nk - 1] - (double)K[nk - 2]);
    }
    i64 j = 0;
    for (i64 r = 0; r < n; r++) {
        const i64 f = rows[r];
        while (j + 1 < nk && K[j + 1] <= f) j++;
        if (K[j] == f) { out[r] = vk[j]; continue; }
        i64 jj = j; if (jj > nk - 2) jj = nk - 2;
        const double t = (double)(f - K[jj]) / (double)(K[jj + 1] - K[jj]);
        const double a = vk[jj], b = vk[jj + 1];
        if (use_cubic) {
            const double h = (double)K[jj + 1] - (double)K[jj];
            const double t2 = t * t, t3 = t2 * t;
            const double c0 = 2 * t3 - 3 * t2 + 1, c1 = t3 - 2 * t2 + t, c2 = -2 * t3 + 3 * t2, c3 = t3 - t2;
            out[r] = c0 * a + c1 * h * m[jj] + c2 * b + c3 * h * m[jj + 1];
        } else {
            out[r] = a + t * (b - a);
        }
    }
}

/* the decoded local transform of bone b at the frames rows[0..n): the track components from the interpolated
 * vectors vx / vy / vz (rotation: rot.exp_map then float32; translation: float32), the other components from the
 * committed state (a copy of S.cur_local[:, b], as make_nk / make_local_c build it).  sc: 4 n doubles of scratch.
 * exp_map: theta = norm (numpy's add.reduce over 3: ((x x + y y) + z z)), half = 0.5 theta,
 * k = theta > 1e-12 ? sin(half) / max(theta, 1e-30) : 0.5, q = (v k, cos(half)). */
static void local_rows(const ek_state *S, const ek_track *T, i64 n, const i64 *rows, const double *vx, const double *vy,
                       const double *vz, float *loc, double *sc)
{
    const int B = S->B, b = T->b;
    for (i64 r = 0; r < n; r++) memcpy(loc + 7 * r, S->cur_local + ((size_t)rows[r] * B + b) * 7, 7 * sizeof(float));
    if (T->kind == 0) {
        double *theta = sc, *half = sc + n, *sn = sc + 2 * n, *cs = sc + 3 * n;
        for (i64 r = 0; r < n; r++) {
            double s = 0.0;
            s += vx[r] * vx[r]; s += vy[r] * vy[r]; s += vz[r] * vz[r];
            theta[r] = sqrt(s);
            half[r] = 0.5 * theta[r];
        }
        ek_sincos(n, half, sn, cs);
        for (i64 r = 0; r < n; r++) {
            const double th = theta[r];
            const double den = th < 1e-30 ? 1e-30 : th;
            const double k = th > 1e-12 ? sn[r] / den : 0.5;
            float *o = loc + 7 * r;
            o[0] = (float)(vx[r] * k); o[1] = (float)(vy[r] * k); o[2] = (float)(vz[r] * k); o[3] = (float)cs[r];
        }
    } else {
        for (i64 r = 0; r < n; r++) {
            float *o = loc + 7 * r;
            o[4] = (float)vx[r]; o[5] = (float)vy[r]; o[6] = (float)vz[r];
        }
    }
}

/* numpy argsort(kind='stable') of v[0..n) (merge sort; ties keep index order) */
static void ek_stable_order(i64 n, const double *v, i64 *ord, i64 *tmp)
{
    for (i64 i = 0; i < n; i++) ord[i] = i;
    for (i64 width = 1; width < n; width *= 2) {
        for (i64 lo = 0; lo < n; lo += 2 * width) {
            i64 mid = lo + width < n ? lo + width : n, hi = lo + 2 * width < n ? lo + 2 * width : n;
            i64 a = lo, b = mid, k = lo;
            while (a < mid && b < hi) { if (v[ord[b]] < v[ord[a]]) tmp[k++] = ord[b++]; else tmp[k++] = ord[a++]; }
            while (a < mid) tmp[k++] = ord[a++];
            while (b < hi) tmp[k++] = ord[b++];
        }
        for (i64 i = 0; i < n; i++) ord[i] = tmp[i];
    }
}

/* the frames of the candidate windows (keys.windows; clip = 1 clips the reach to the first / last key):
 * rows strictly between K[pos - reach] and K[pos + reach] per candidate, in candidate order.  Returns the row count. */
static i64 make_windows(const i64 *K, i64 nK, const i64 *pos, i64 np_, int reach, int clip, i64 *idx, int *win)
{
    i64 m = 0;
    for (i64 w = 0; w < np_; w++) {
        i64 a = pos[w] - reach, b = pos[w] + reach;
        if (clip) { if (a < 0) a = 0; if (b > nK - 1) b = nK - 1; }
        for (i64 f = K[a] + 1; f < K[b]; f++) { idx[m] = f; win[m] = (int)w; m++; }
    }
    return m;
}

/* one column of the Catmull-Rom interpolation of the integer keys nk (column c of an [nk, 3] array, dequantized
 * with step s) at the frames rows[0..n) */
static void interp_keys_col(i64 nk, const i64 *K, const i64 *nkv, int c, double s, i64 n, const i64 *rows, double *vk,
                            double *m, double *out)
{
    for (i64 k = 0; k < nk; k++) vk[k] = dq(nkv[3 * k + c], s);
    interp_col(nk, K, vk, n, rows, 1, m, out);
}

/* scratch buffers of the key loops */
static ek_buf g_b[48];
#define GB(i, type, count) ((type *)bget(&g_b[i], sizeof(type) * (size_t)(count)))

/* the committed object rows of bone b's parent, AoS [F][7] (rot x y z w, pos x y z; constant while one track is worked
 * on), or NULL for a root */
static const double *gather_parent(const ek_state *S, const ek_track *T, double *P)
{
    const int pa = T->par_abs[0], F = S->F, B = S->B;
    if (pa < 0) return 0;
    for (int f = 0; f < F; f++) {
        const double *prot = S->obj_rot + ((size_t)f * B + pa) * 4, *ppos = S->obj_pos + ((size_t)f * B + pa) * 3;
        double *o = P + 7 * (size_t)f;
        o[0] = prot[0]; o[1] = prot[1]; o[2] = prot[2]; o[3] = prot[3]; o[4] = ppos[0]; o[5] = ppos[1]; o[6] = ppos[2];
    }
    return P;
}

/* the committed locals of the subtree bones 1 .. nsub-1 (bone b itself is the candidate), bone-major AoS [nsub][F][7]
 * float32 (constant while one track is worked on: its rounds only change bone b) */
static const float *gather_locals(const ek_state *S, const ek_track *T, float *Lm)
{
    const int F = S->F, B = S->B, ns = T->nsub;
    if (ns <= 1) return Lm;
    for (int f = 0; f < F; f++) {
        const float *row = S->cur_local + (size_t)f * B * 7;
        for (int j = 1; j < ns; j++) memcpy(Lm + ((size_t)j * F + f) * 7, row + (size_t)T->sub[j] * 7, 7 * sizeof(float));
    }
    return Lm;
}

/* ev_rows with contiguous sources: the committed locals from Lm (bone-major AoS, or the frame-major state when NULL),
 * bone b's parent rows from P (AoS [F][7], or the state), the raw rows from S->raw_a (AoS [B][F][12]); per bone the
 * active rows are gathered into structure-of-arrays temporaries (one frame's inputs are three contiguous records),
 * computed with the vectorised stages and scattered into R / E -- the same per-element expressions as ev_rows. */
static int ev_rows2(const ek_state *S, const ek_track *T, i64 n, const i64 *idx, const float *loc, const int *win, i64 nwin,
                    const uint8_t *wact, const double *glim, uint8_t *wbad, double *R, double *E, const float *Lm, const double *P,
                    ek_buf *bufA, ek_buf *bufT)
{
    const int B = S->B, F = S->F;
    const size_t N = (size_t)n;
    i64 *act = (i64 *)bget(bufA, sizeof(i64) * (N + 1));
    double *tmp = (double *)bget(bufT, sizeof(double) * (N * 27 + 8));
    if (!act || !tmp) return -2;
    i64 na = 0;
    for (i64 r = 0; r < n; r++) if (wact[win[r]]) act[na++] = r;
    if (glim) for (i64 w = 0; w < nwin; w++) wbad[w] = 0;
    for (int j = 0; j < T->nsub && na > 0; j++) {
        const int c = T->sub[j], pa = T->par_abs[j], prel = T->par_rel[j];
        const size_t NA = (size_t)na;
        double *q0 = tmp, *q1 = q0 + NA, *q2 = q1 + NA, *q3 = q2 + NA, *t0 = q3 + NA, *t1 = t0 + NA, *t2 = t1 + NA;
        double *p0 = t2 + NA, *p1 = p0 + NA, *p2 = p1 + NA, *p3 = p2 + NA, *s0 = p3 + NA, *s1 = s0 + NA, *s2 = s1 + NA;
        double *rc = s2 + NA, *rp = rc + 9 * NA, *er = rp + 3 * NA;
        const float *Lj = (j > 0 && Lm) ? Lm + (size_t)j * F * 7 : 0;
        const double *RA = S->raw_a + (size_t)c * F * 12;
        for (i64 i = 0; i < na; i++) {
            const i64 r = act[i], f = idx[r];
            const float *lf = (j == 0) ? loc + 7 * (size_t)r : (Lj ? Lj + 7 * (size_t)f : S->cur_local + ((size_t)f * B + c) * 7);
            q0[i] = (double)lf[0]; q1[i] = (double)lf[1]; q2[i] = (double)lf[2]; q3[i] = (double)lf[3];
            t0[i] = (double)lf[4]; t1[i] = (double)lf[5]; t2[i] = (double)lf[6];
            const double *ra = RA + 12 * (size_t)f;
            rc[i] = ra[0]; rc[NA + i] = ra[1]; rc[2 * NA + i] = ra[2]; rc[3 * NA + i] = ra[3]; rc[4 * NA + i] = ra[4];
            rc[5 * NA + i] = ra[5]; rc[6 * NA + i] = ra[6]; rc[7 * NA + i] = ra[7]; rc[8 * NA + i] = ra[8];
            rp[i] = ra[9]; rp[NA + i] = ra[10]; rp[2 * NA + i] = ra[11];
        }
        if (pa < 0) {
            st_norm((int)na, q0, q1, q2, q3);
        } else {
            if (j > 0 && prel >= 0) {
                const double *PR = R + (size_t)prel * 7 * N;
                for (i64 i = 0; i < na; i++) {
                    const i64 r = act[i];
                    p0[i] = PR[r]; p1[i] = PR[N + r]; p2[i] = PR[2 * N + r]; p3[i] = PR[3 * N + r];
                    s0[i] = PR[4 * N + r]; s1[i] = PR[5 * N + r]; s2[i] = PR[6 * N + r];
                }
            } else if (j == 0 && P) {
                for (i64 i = 0; i < na; i++) {
                    const double *o = P + 7 * (size_t)idx[act[i]];
                    p0[i] = o[0]; p1[i] = o[1]; p2[i] = o[2]; p3[i] = o[3]; s0[i] = o[4]; s1[i] = o[5]; s2[i] = o[6];
                }
            } else {
                for (i64 i = 0; i < na; i++) {
                    const i64 f = idx[act[i]];
                    const double *prot = S->obj_rot + ((size_t)f * B + pa) * 4, *ppos = S->obj_pos + ((size_t)f * B + pa) * 3;
                    p0[i] = prot[0]; p1[i] = prot[1]; p2[i] = prot[2]; p3[i] = prot[3];
                    s0[i] = ppos[0]; s1[i] = ppos[1]; s2[i] = ppos[2];
                }
            }
            st_fk((int)na, q0, q1, q2, q3, t0, t1, t2, p0, p1, p2, p3, s0, s1, s2);
        }
        st_err((int)na, q0, q1, q2, q3, t0, t1, t2, rc, rc + NA, rc + 2 * NA, rc + 3 * NA, rc + 4 * NA, rc + 5 * NA, rc + 6 * NA,
               rc + 7 * NA, rc + 8 * NA, rp, rp + NA, rp + 2 * NA, S->shell, er);
        double *RJ = R + (size_t)j * 7 * N, *EJ = E + (size_t)j * N;
        int nan_seen = 0, newbad = 0;
        for (i64 i = 0; i < na; i++) {
            const i64 r = act[i];
            RJ[r] = q0[i]; RJ[N + r] = q1[i]; RJ[2 * N + r] = q2[i]; RJ[3 * N + r] = q3[i];
            RJ[4 * N + r] = t0[i]; RJ[5 * N + r] = t1[i]; RJ[6 * N + r] = t2[i];
            const double e = er[i];
            EJ[r] = e;
            if (e != e) nan_seen = 1;
            if (glim && e > glim[j] && !wbad[win[r]]) { wbad[win[r]] = 1; newbad = 1; }
        }
        if (nan_seen) return -1;
        if (newbad) {
            i64 k = 0;
            for (i64 i = 0; i < na; i++) if (!wbad[win[act[i]]]) act[k++] = act[i];
            na = k;
        }
    }
    return 0;
}

/* the local rows of a per-component track: component c interpolated on its own key set Ks[c] (values kv[c],
 * dequantized with step[c]) at the frames rows[0..n) -- keys.reconstruct_c at those rows */
static void local_rows_c(const ek_state *S, const ek_track *T, i64 *const Ks[3], const i64 nKs[3], i64 *const kv[3], const double *step,
                         i64 n, const i64 *rows, double *vk, double *mm, double *vx, double *sc, float *loc)
{
    for (int c = 0; c < 3; c++) {
        for (i64 k = 0; k < nKs[c]; k++) vk[k] = dq(kv[c][k], step[c]);
        interp_col(nKs[c], Ks[c], vk, n, rows, 1, mm, vx + (size_t)c * n);
    }
    local_rows(S, T, n, rows, vx, vx + n, vx + 2 * n, loc, sc);
}

/* write the accepted windows' rows into the state: cur_local[f, b] = loc row, obj_rot / obj_pos / err of the subtree */
static void commit_rows(ek_state *S, const ek_track *T, i64 ni, const i64 *idx, const int *win, const uint8_t *acc,
                        const float *loc, const double *R, const double *E)
{
    const int B = S->B;
    for (i64 r = 0; r < ni; r++) {
        if (!acc[win[r]]) continue;
        const i64 f = idx[r];
        memcpy(S->cur_local + ((size_t)f * B + T->b) * 7, loc + 7 * (size_t)r, 7 * sizeof(float));
        for (int j = 0; j < T->nsub; j++) {
            const int c = T->sub[j];
            const double *RJ = R + (size_t)j * 7 * ni;
            double *orr = S->obj_rot + ((size_t)f * B + c) * 4, *opp = S->obj_pos + ((size_t)f * B + c) * 3;
            orr[0] = RJ[r]; orr[1] = RJ[ni + r]; orr[2] = RJ[2 * ni + r]; orr[3] = RJ[3 * ni + r];
            opp[0] = RJ[4 * ni + r]; opp[1] = RJ[5 * ni + r]; opp[2] = RJ[6 * ni + r];
            S->err[(size_t)f * B + c] = E[(size_t)j * ni + r];
        }
    }
}

/* ============================================================================================================== */
/* the rate-distortion key removal                                                                                  */
/* ============================================================================================================== */
/* D [np, nsub]: per window and subtree bone the sum over its rows of (E - committed error), np.bincount order */
static void rd_D(const ek_state *S, const ek_track *T, i64 ni, const i64 *idx, const int *win, i64 np_, const uint8_t *bad,
                 const double *E, double *D)
{
    const int B = S->B, nsub = T->nsub;
    for (i64 k = 0; k < np_ * nsub; k++) D[k] = 0.0;
    for (i64 r = 0; r < ni; r++) {
        const int w = win[r];
        if (bad[w]) continue;
        const i64 f = idx[r];
        double *d = D + (size_t)w * nsub;
        for (int j = 0; j < nsub; j++) d[j] += E[(size_t)j * ni + r] - S->err[(size_t)f * B + T->sub[j]];
    }
}

/* the d2t residual of key k (k >= 2) of one integer column x (stride 3 for [n, 3] arrays): predict.key_residuals with
 * pred 2: d_k - rint(d_{k-1} * g_k / g_{k-1}) */
static inline i64 d2t_at(const i64 *K, const i64 *x, int stride, i64 k)
{
    const i64 d1 = x[stride * k] - x[stride * (k - 1)], d0 = x[stride * (k - 1)] - x[stride * (k - 2)];
    const double ratio = (double)(K[k] - K[k - 1]) / (double)(K[k - 1] - K[k - 2]);
    return d1 - (i64)nearbyint((double)d0 * ratio);
}

/* the acceptance in slope order (keys._accept): sl, bad, D -> acc, slope[pos] updates (blocked -> inf); rem_sub / rem_tot updated.
 * Returns the number accepted. */
static i64 accept_rd(const ek_track *T, i64 np_, const i64 *pos, const uint8_t *bad, const double *sl, const double *D, double lam,
                     double *rem_sub, double *rem_tot, double *slope, uint8_t *acc, i64 *ord, i64 *stmp)
{
    const int nsub = T->nsub;
    ek_stable_order(np_, sl, ord, stmp);
    memset(acc, 0, (size_t)np_);
    i64 nacc = 0;
    for (i64 k = 0; k < np_; k++) {
        const i64 w = ord[k];
        if (bad[w]) continue;
        if (sl[w] > lam) break;
        const double *d = D + (size_t)w * nsub;
        int fits = 1;
        for (int j = 0; j < nsub; j++) if (!(d[j] <= rem_sub[j] + 1e-12)) { fits = 0; break; }
        const double ds = np_sum(d, nsub);
        if (fits && ds <= *rem_tot + 1e-12) {
            acc[w] = 1; nacc++;
            for (int j = 0; j < nsub; j++) rem_sub[j] = rem_sub[j] - d[j];
            *rem_tot -= ds;
        } else {
            slope[pos[w]] = INFINITY;
        }
    }
    return nacc;
}

/* after an accepting round: S.esum[sub] += D[acc].sum(axis=0) (numpy: pairwise when the subtree is one bone, else row
 * by row), S.emax[sub] = max(S.emax[sub], E[accepted rows].max(axis=0)) */
static void rd_esum_emax(ek_state *S, const ek_track *T, i64 np_, const uint8_t *acc, const double *D, i64 ni, const int *win,
                         const double *E, double *tmp)
{
    const int nsub = T->nsub;
    if (nsub == 1) {
        i64 m = 0;
        for (i64 w = 0; w < np_; w++) if (acc[w]) tmp[m++] = D[w];
        S->esum[T->sub[0]] += np_sum(tmp, m);
    } else {
        int first = 1;
        for (i64 w = 0; w < np_; w++) if (acc[w]) {
            const double *d = D + (size_t)w * nsub;
            if (first) { for (int j = 0; j < nsub; j++) tmp[j] = d[j]; first = 0; }
            else for (int j = 0; j < nsub; j++) tmp[j] += d[j];
        }
        for (int j = 0; j < nsub; j++) S->esum[T->sub[j]] += tmp[j];
    }
    for (int j = 0; j < nsub; j++) {
        const int c = T->sub[j];
        double mx = 0.0; int any = 0;
        const double *EJ = E + (size_t)j * ni;
        for (i64 r = 0; r < ni; r++) if (acc[win[r]]) { if (!any || EJ[r] > mx) mx = EJ[r]; any = 1; }
        if (any && mx > S->emax[c]) S->emax[c] = mx;       /* np.maximum (no NaN in a committed state) */
    }
}

/* ------------------------------------------------------------------------------------------------------------ */
/* keys.remove_keys: the RD removal of one track's keys at one lambda (off: per-key integer offsets, zeros here).  */
/*   K [*nK], off [*nK, 3], dirty [*nK] u8, slope [*nK]: in / out (*nK updated); cap_sub [nsub] = S.mlim[sub] * F;  */
/*   total = S.total; st [5] += (rounds, evals, cols, frames, n_acc); flag[0] = 1 when a round accepted.            */
/* Returns 0, 1 (a NaN: stopped at the start of the round, nothing of it applied), -2 (allocation).               */
/* ------------------------------------------------------------------------------------------------------------ */
int ek_rk_rd(ek_state *S, const ek_track *T, i64 *K, i64 *off, uint8_t *dirty, double *slope, i64 *nK_io, const double *step,
             double lam, int reach, int modulus, int clip, const double *cap_sub, double total, double *st, i64 *flag)
{
    const int F = S->F, B = S->B, nsub = T->nsub;
    const size_t NF = (size_t)F;
    i64 nK = *nK_io;
    int ret = 0;
    double r_rounds = 0;
    i64 *pos = GB(0, i64, NF), *K2 = GB(1, i64, NF), *nk2 = GB(2, i64, 3 * NF), *nkK = GB(3, i64, 3 * NF), *idx = GB(5, i64, NF);
    i64 *ord = GB(8, i64, NF), *stmp = GB(10, i64, NF);
    int *win = GB(11, int, NF);
    uint8_t *keep = GB(12, uint8_t, NF), *wact = GB(13, uint8_t, NF), *bad = GB(14, uint8_t, NF), *acc = GB(19, uint8_t, NF);
    double *sl = GB(20, double, NF), *lx = GB(21, double, 3 * NF), *ly = GB(22, double, 3 * NF), *vk = GB(23, double, NF), *mm = GB(24, double, NF);
    double *vx = GB(25, double, 3 * NF), *sc = GB(26, double, 4 * NF), *D = GB(28, double, NF * nsub + 16);
    float *loc = GB(29, float, 7 * NF);
    double *R = GB(31, double, 7 * NF * nsub), *E = GB(32, double, NF * nsub), *glim = GB(35, double, nsub + 1);
    double *rem = GB(36, double, B + nsub + NF + 1);
    if (!pos || !K2 || !nk2 || !nkK || !idx || !ord || !stmp || !win || !keep || !wact || !bad || !acc || !sl || !lx || !ly || !vk || !mm ||
        !vx || !sc || !D || !loc || !R || !E || !glim || !rem) { ret = -2; goto done; }
    for (int j = 0; j < nsub; j++) glim[j] = S->limits[T->sub[j]];
    double *Pbuf = GB(39, double, 7 * NF);
    float *Lbuf = GB(46, float, 7 * NF * (size_t)nsub);
    if (!Pbuf || !Lbuf) { ret = -2; goto done; }
    const double *P = 0;
    const float *Lm = 0;
    int have_p = 0;
    while (nK > 2 * reach) {
        i64 np_ = 0, last = -modulus;
        for (i64 k = reach; k < nK - reach; k++)
            if ((dirty[k] || slope[k] <= lam) && k - last >= modulus) { pos[np_++] = k; last = k; }
        if (np_ == 0) break;
        memset(keep, 1, (size_t)nK);
        for (i64 w = 0; w < np_; w++) keep[pos[w]] = 0;
        i64 nK2 = 0;
        for (i64 k = 0; k < nK; k++) {
            for (int c = 0; c < 3; c++) nkK[3 * k + c] = q_int(T->v[3 * (size_t)K[k] + c], step[c]) + off[3 * k + c];
            if (keep[k]) { K2[nK2] = K[k]; for (int c = 0; c < 3; c++) nk2[3 * nK2 + c] = nkK[3 * k + c]; nK2++; }
        }
        const i64 ni = make_windows(K, nK, pos, np_, reach, clip, idx, win);
        for (int c = 0; c < 3; c++) interp_keys_col(nK2, K2, nk2, c, step[c], ni, idx, vk, mm, vx + (size_t)c * ni);
        local_rows(S, T, ni, idx, vx, vx + ni, vx + 2 * ni, loc, sc);
        memset(wact, 1, (size_t)np_);
        if (!have_p && ni > F / 8) { P = gather_parent(S, T, Pbuf); Lm = gather_locals(S, T, Lbuf); have_p = 1; }   /* (big rounds) */
        const int e = ev_rows2(S, T, ni, idx, loc, win, np_, wact, glim, bad, R, E, Lm, P, &g_b[37], &g_b[38]);
        if (e < 0) { ret = e == -1 ? 1 : e; goto done; }
        r_rounds += 1;
        st[1] += 1; st[2] += nsub; st[3] += (double)nsub * (double)ni;
        rd_D(S, T, ni, idx, win, np_, bad, E, D);
        /* the key-bit proxy of the candidates on the current keys: sum_c log2(1 + 2 |d2t residual|) + 1.5 */
        for (i64 w = 0; w < np_; w++) for (int c = 0; c < 3; c++) {
            const i64 r = d2t_at(K, nkK + c, 3, pos[w]);
            lx[3 * w + c] = 1.0 + 2.0 * (double)(r < 0 ? -r : r);
        }
        ek_log2(3 * np_, lx, ly);
        for (i64 w = 0; w < np_; w++) {
            double bsum = 0.0;
            bsum += ly[3 * w]; bsum += ly[3 * w + 1]; bsum += ly[3 * w + 2];
            const double bits = bsum + 1.5;
            const double ds = np_sum(D + (size_t)w * nsub, nsub);
            sl[w] = ds / (bits > 0.5 ? bits : 0.5);
            slope[pos[w]] = sl[w]; dirty[pos[w]] = 0;
        }
        for (i64 w = 0; w < np_; w++) if (bad[w]) slope[pos[w]] = INFINITY;
        double *rem_sub = rem, *tmp = rem + nsub;
        for (int j = 0; j < nsub; j++) rem_sub[j] = cap_sub[j] - S->esum[T->sub[j]];
        double rem_tot = total - np_sum(S->esum, B);
        const i64 nacc = accept_rd(T, np_, pos, bad, sl, D, lam, rem_sub, &rem_tot, slope, acc, ord, stmp);
        if (nacc) {
            commit_rows(S, T, ni, idx, win, acc, loc, R, E);
            rd_esum_emax(S, T, np_, acc, D, ni, win, E, tmp);
            st[4] += (double)nacc;
            for (i64 w = 0; w < np_; w++) if (acc[w]) for (int r = 1; r <= reach; r++) {
                i64 lo = pos[w] - r, hi = pos[w] + r;
                if (lo < 0) lo = 0; if (hi > nK - 1) hi = nK - 1;
                dirty[lo] = 1; dirty[hi] = 1;
            }
            memset(keep, 1, (size_t)nK);
            for (i64 w = 0; w < np_; w++) if (acc[w]) keep[pos[w]] = 0;
            i64 k2 = 0;
            for (i64 k = 0; k < nK; k++) if (keep[k]) {
                K[k2] = K[k]; dirty[k2] = dirty[k]; slope[k2] = slope[k];
                off[3 * k2] = off[3 * k]; off[3 * k2 + 1] = off[3 * k + 1]; off[3 * k2 + 2] = off[3 * k + 2];
                k2++;
            }
            nK = k2;
            flag[0] = 1;
        }
    }
done:
    st[0] += r_rounds;
    *nK_io = nK;
    return ret;
}

/* ------------------------------------------------------------------------------------------------------------ */
/* keys.ladder_c with keys.remove_keys_c: the lambda ladder over the per-component RD states, as one call.          */
/* ------------------------------------------------------------------------------------------------------------ */
typedef struct { i64 *K, *V; uint8_t *D; double *SL; i64 n; } ek_comp;

/* is frame f a key of the sorted set K[0..n) */
static inline int in_sorted(const i64 *K, i64 n, i64 f)
{
    i64 lo = 0, hi = n;
    while (lo < hi) { const i64 m = (lo + hi) >> 1; if (K[m] < f) lo = m + 1; else hi = m; }
    return lo < n && K[lo] == f;
}

/* remove_keys_rd_c for component c of track T (st [5] += rounds, evals, cols, frames, n_acc) */
static int rd_c_call(ek_state *S, const ek_track *T, int c, ek_comp *cp, const double *step, double lam, int reach, int modulus,
                     int price_flags, double gap_bits, double flag_const, double total, double *st)
{
    const int F = S->F, B = S->B, nsub = T->nsub;
    const size_t NF = (size_t)F;
    i64 *pos = GB(0, i64, NF), *Kr = GB(1, i64, NF), *Vr = GB(2, i64, NF), *nkq = GB(3, i64, NF), *idx = GB(5, i64, NF);
    i64 *ord = GB(8, i64, NF), *stmp = GB(10, i64, NF), *cs = GB(9, i64, NF + 1);
    int *win = GB(11, int, NF);
    uint8_t *keep = GB(12, uint8_t, NF), *wact = GB(13, uint8_t, NF), *bad = GB(14, uint8_t, NF), *acc = GB(19, uint8_t, NF);
    uint8_t *chg = GB(15, uint8_t, NF);
    double *sl = GB(20, double, NF), *lx = GB(21, double, NF), *ly = GB(22, double, NF), *vk = GB(23, double, NF), *mm = GB(24, double, NF);
    double *vx = GB(25, double, 3 * NF), *sc = GB(26, double, 4 * NF), *D = GB(28, double, NF * nsub + 16);
    float *loc = GB(29, float, 7 * NF);
    double *R = GB(31, double, 7 * NF * nsub), *E = GB(32, double, NF * nsub), *glim = GB(35, double, nsub + 1);
    double *rem = GB(36, double, B + 2 * nsub + NF + 1);
    if (!pos || !Kr || !Vr || !nkq || !idx || !ord || !stmp || !cs || !win || !keep || !wact || !bad || !acc || !chg || !sl || !lx ||
        !ly || !vk || !mm || !vx || !sc || !D || !loc || !R || !E || !glim || !rem) return -2;
    for (int j = 0; j < nsub; j++) glim[j] = S->limits[T->sub[j]];
    double *cap_sub = rem + B, *rem_sub = cap_sub + nsub, *tmp = rem_sub + nsub;
    for (int j = 0; j < nsub; j++) cap_sub[j] = S->mlim[T->sub[j]] * (double)F;
    ek_comp *me = cp + c;
    double r_rounds = 0;
    int ret = 0;
    while (me->n > 2 * reach) {
        const i64 nK = me->n;
        i64 *K = me->K;
        i64 np_ = 0, last = -modulus;
        for (i64 k = reach; k < nK - reach; k++)
            if ((me->D[k] || me->SL[k] <= lam) && k - last >= modulus) { pos[np_++] = k; last = k; }
        if (np_ == 0) break;
        memset(keep, 1, (size_t)nK);
        for (i64 w = 0; w < np_; w++) keep[pos[w]] = 0;
        i64 nr = 0;
        for (i64 k = 0; k < nK; k++) if (keep[k]) { Kr[nr] = K[k]; Vr[nr] = me->V[k]; nr++; }
        const i64 ni = make_windows(K, nK, pos, np_, reach, 0, idx, win);
        i64 *Ks2[3] = {cp[0].K, cp[1].K, cp[2].K}, *kv2[3] = {cp[0].V, cp[1].V, cp[2].V}, n2[3] = {cp[0].n, cp[1].n, cp[2].n};
        Ks2[c] = Kr; kv2[c] = Vr; n2[c] = nr;
        local_rows_c(S, T, Ks2, n2, kv2, step, ni, idx, vk, mm, vx, sc, loc);
        memset(wact, 1, (size_t)np_);
        const int e = ev_rows2(S, T, ni, idx, loc, win, np_, wact, glim, bad, R, E, 0, 0, &g_b[37], &g_b[38]);
        if (e < 0) { ret = e == -1 ? 1 : e; break; }
        r_rounds += 1;
        st[1] += 1; st[2] += nsub; st[3] += (double)nsub * (double)ni;
        rd_D(S, T, ni, idx, win, np_, bad, E, D);
        /* bits: log2(1 + 2 |d2t residual|) of the re-quantized raw component at the candidate + its gap / flags price */
        for (i64 k = 0; k < nK; k++) nkq[k] = q_int(T->v[3 * (size_t)K[k] + c], step[c]);
        for (i64 w = 0; w < np_; w++) { const i64 r = d2t_at(K, nkq, 1, pos[w]); lx[w] = 1.0 + 2.0 * (double)(r < 0 ? -r : r); }
        ek_log2(np_, lx, ly);
        for (i64 w = 0; w < np_; w++) {
            double bits;
            if (price_flags) {
                const i64 f = K[pos[w]];
                int other = 0;
                for (int c2 = 0; c2 < 3 && !other; c2++) if (c2 != c && in_sorted(cp[c2].K, cp[c2].n, f)) other = 1;
                bits = (ly[w] + 0.0) + (other ? 0.0 : flag_const);
            } else {
                bits = ly[w] + gap_bits;
            }
            const double ds = np_sum(D + (size_t)w * nsub, nsub);
            sl[w] = ds / (bits > 0.5 ? bits : 0.5);
            me->SL[pos[w]] = sl[w]; me->D[pos[w]] = 0;
        }
        for (i64 w = 0; w < np_; w++) if (bad[w]) me->SL[pos[w]] = INFINITY;
        for (int j = 0; j < nsub; j++) rem_sub[j] = cap_sub[j] - S->esum[T->sub[j]];
        double rem_tot = total - np_sum(S->esum, B);
        const i64 nacc = accept_rd(T, np_, pos, bad, sl, D, lam, rem_sub, &rem_tot, me->SL, acc, ord, stmp);
        if (nacc) {
            commit_rows(S, T, ni, idx, win, acc, loc, R, E);
            rd_esum_emax(S, T, np_, acc, D, ni, win, E, tmp);
            st[4] += (double)nacc;
            for (i64 w = 0; w < np_; w++) if (acc[w]) for (int r = 1; r <= reach; r++) {
                i64 lo = pos[w] - r, hi = pos[w] + r;
                if (lo < 0) lo = 0; if (hi > nK - 1) hi = nK - 1;
                me->D[lo] = 1; me->D[hi] = 1;
            }
            memset(keep, 1, (size_t)nK);
            for (i64 w = 0; w < np_; w++) if (acc[w]) keep[pos[w]] = 0;
            i64 k2 = 0;
            for (i64 k = 0; k < nK; k++) if (keep[k]) { K[k2] = K[k]; me->V[k2] = me->V[k]; me->D[k2] = me->D[k]; me->SL[k2] = me->SL[k]; k2++; }
            me->n = k2;
            /* _touch_other_components: the other components' keys whose windows contain a changed frame get dirty */
            memset(chg, 0, NF);
            for (i64 r = 0; r < ni; r++) if (acc[win[r]]) chg[idx[r]] = 1;
            cs[0] = 0;
            for (int f = 0; f < F; f++) cs[f + 1] = cs[f] + chg[f];
            for (int c2 = 0; c2 < 3; c2++) {
                if (c2 == c) continue;
                ek_comp *o = cp + c2;
                if (o->n <= 2 * reach) continue;
                for (i64 p = reach; p < o->n - reach; p++) {
                    const i64 lo = o->K[p - reach] + 1, hi = o->K[p + reach];
                    if (cs[hi] - cs[lo] > 0) o->D[p] = 1;
                }
            }
        }
    }
    st[0] += r_rounds;
    return ret;
}

/*   T [ntr] tracks, steps [ntr, 3]; comps: 3 ntr ek_comp (in / out); ord0 / ord1 [ntr] the two track orders;
 *   total = S.total; st [7] += (rounds, evals, cols, frames, n_acc), st[5] = levels, st[6] = lam_final / m_p.
 * Returns 0, 1 (a NaN at the start of a round of component c of the i-th track of a level: nothing of that round
 * applied; where [4] = (level, i, c, lam) -- the caller finishes the ladder with numpy from there), -2. */
int ek_ladder_c(ek_state *S, const ek_track *T, int ntr, const double *steps, ek_comp *comps, const i64 *ord0, const i64 *ord1,
                double lam0, double ratio, int max_levels, double m_p, int reach, int price_flags, double gap_bits, double flag_const,
                double total, double *st, double *where)
{
    const int modulus = 2 * reach;
    double lam = lam0 * m_p;
    for (int level = 0; level < max_levels; level++) {
        st[5] = level + 1; st[6] = lam / m_p;
        const i64 *ord = (level % 2) ? ord1 : ord0;
        for (int i = 0; i < ntr; i++) {
            const int t = (int)ord[i];
            for (int c = 0; c < 3; c++) {
                const int e = rd_c_call(S, T + t, c, comps + 3 * t, steps + 3 * t, lam, reach, modulus, price_flags, gap_bits, flag_const,
                                        total, st);
                if (e != 0) {                                  /* where the numpy continuation resumes */
                    where[0] = level; where[1] = i; where[2] = c; where[3] = lam;
                    return e == -1 ? 1 : e;
                }
            }
        }
        double nxt = INFINITY;
        int pending = 0;
        for (int t = 0; t < ntr; t++) for (int c = 0; c < 3; c++) {
            const ek_comp *cp = comps + 3 * t + c;
            const i64 hi = cp->n - reach > reach ? cp->n - reach : reach;
            for (i64 k = reach; k < hi; k++) {
                const double v = cp->SL[k];
                if (v > lam && v < nxt) nxt = v;
                if (cp->D[k]) pending = 1;
            }
        }
        if (!isfinite(nxt) && !pending) break;
        if (total - np_sum(S->esum, S->B) <= 1e-6 * total) break;
        if (isfinite(nxt)) { const double a = lam * ratio; lam = a >= nxt ? a : nxt; } else lam = lam * ratio;
    }
    return 0;
}

/* ------------------------------------------------------------------------------------------------------------ */
/* keys.reconstruct(K, nk, step, F, kind) (cubic = 1; 0 = linear): the decoded track of all frames from its key    */
/* integers nk [nK, 3]: dequantize (float32 of n * step), interp_vectors, rot.exp_map (rotation), float32.          */
/* out: rotation [F, 4] / translation [F, 3] float32.                                                               */
/* ------------------------------------------------------------------------------------------------------------ */
int ek_reconstruct(const i64 *K, i64 nK, const i64 *nk, const double *step, i64 F, int kind, int cubic, float *out)
{
    const size_t NF = (size_t)F;
    i64 *rows = (i64 *)bget(&g_k[2], sizeof(i64) * NF);
    double *vx = (double *)bget(&g_k[3], sizeof(double) * (7 * NF + 2 * (size_t)nK + 8));
    if (!rows || !vx) return -2;
    double *sc = vx + 3 * NF, *vk = sc + 4 * NF, *mm = vk + nK;
    for (i64 f = 0; f < F; f++) rows[f] = f;
    for (int c = 0; c < 3; c++) {
        for (i64 k = 0; k < nK; k++) vk[k] = dq(nk[3 * k + c], step[c]);
        interp_col(nK, K, vk, F, rows, cubic, mm, vx + (size_t)c * NF);
    }
    const double *x = vx, *y = vx + NF, *z = vx + 2 * NF;
    if (kind == 0) {
        double *theta = sc, *half = sc + NF, *sn = sc + 2 * NF, *cs = sc + 3 * NF;
        for (i64 r = 0; r < F; r++) {
            double s = 0.0;
            s += x[r] * x[r]; s += y[r] * y[r]; s += z[r] * z[r];
            theta[r] = sqrt(s); half[r] = 0.5 * theta[r];
        }
        ek_sincos(F, half, sn, cs);
        for (i64 r = 0; r < F; r++) {
            const double th = theta[r], den = th < 1e-30 ? 1e-30 : th;
            const double k = th > 1e-12 ? sn[r] / den : 0.5;
            float *o = out + 4 * (size_t)r;
            o[0] = (float)(x[r] * k); o[1] = (float)(y[r] * k); o[2] = (float)(z[r] * k); o[3] = (float)cs[r];
        }
    } else {
        for (i64 r = 0; r < F; r++) { float *o = out + 3 * (size_t)r; o[0] = (float)x[r]; o[1] = (float)y[r]; o[2] = (float)z[r]; }
    }
    return 0;
}

/* ============================================================================================================== */
/* Sessions: a SESSION holds one track's subtree with the committed state gathered once (bone-major float64 rows) */
/* and evaluates candidate steps of that track with it. The Python driver (kernels.grow_step) opens a session only  */
/* for a stretch of candidates between two commits (the committed state cannot change inside it), so the gathered   */
/* rows are exact.                                                                                                   */
/* ============================================================================================================== */
typedef struct {
    int open, F, nsub, have_raw, kind, b, finite, fexit;
    ek_state S;
    ek_track T;
    float *L;              /* [nsub][F][7] committed locals of the subtree bones (the float32 state, one AoS block per bone) */
    int g;                 /* bones [0, g) of the subtree have their block gathered (lazily, 8 at a time, on first use) */
    double *P, *PR;        /* [7][F] parent rows of bone 0: committed state / raw chain (parent_raw) */
    double *R[2], *E[2];   /* candidate outputs, two slots (cur / kept): [nsub][7][F] object rot / pos, [nsub][F] errors */
    double *st[2];         /* [nsub][3] per bone max, sum, count */
    float *loc[2];         /* [F][7] the candidate's local transform of bone 0 */
    int nd[2];
    int cur;               /* slot of the candidate being evaluated; 1 - cur = the kept (last passing) one */
    ek_buf buf;
} ek_sess_t;
static ek_sess_t g_ss;

/* open a session on track T of state S (copies of the two descriptors); gathers the subtree's committed locals and the
 * committed parent rows of bone b.  Returns 0 or -2. */
int ek_sess_begin(const ek_state *S, const ek_track *T)
{
    ek_sess_t *s = &g_ss;
    const int F = S->F, B = S->B, ns = T->nsub;
    const size_t NF = (size_t)F;
    const size_t need = sizeof(double) * NF * (7 * (size_t)ns * 3 + 14 + (size_t)ns * 2) + sizeof(double) * 6 * (size_t)ns
                      + sizeof(float) * 14 * NF + 256;
    char *p = (char *)bget(&s->buf, need);
    if (!p) return -2;
    s->S = *S; s->T = *T; s->F = F; s->nsub = ns; s->have_raw = 0; s->kind = T->kind; s->b = T->b;
    s->L = (float *)p; p += sizeof(double) * 7 * NF * ns;   /* (double-sized; the float blocks use half of it) */
    s->P = (double *)p; p += sizeof(double) * 7 * NF;
    s->PR = (double *)p; p += sizeof(double) * 7 * NF;
    for (int k = 0; k < 2; k++) {
        s->R[k] = (double *)p; p += sizeof(double) * 7 * NF * ns;
        s->E[k] = (double *)p; p += sizeof(double) * NF * ns;
        s->st[k] = (double *)p; p += sizeof(double) * 3 * ns;
    }
    for (int k = 0; k < 2; k++) { s->loc[k] = (float *)p; p += sizeof(float) * 7 * NF; }
    s->nd[0] = s->nd[1] = 0; s->cur = 0;
    /* bone b's committed local now; the other subtree bones' blocks when an evaluation first reaches them */
    for (int f = 0; f < F; f++) memcpy(s->L + 7 * (size_t)f, S->cur_local + ((size_t)f * B + T->sub[0]) * 7, 7 * sizeof(float));
    s->g = 1;
    const int pa = T->par_abs[0];
    if (pa >= 0) {
        for (int f = 0; f < F; f++) {
            const double *prot = S->obj_rot + ((size_t)f * B + pa) * 4, *ppos = S->obj_pos + ((size_t)f * B + pa) * 3;
            for (int k = 0; k < 4; k++) s->P[k * NF + f] = prot[k];
            for (int k = 0; k < 3; k++) s->P[(4 + k) * NF + f] = ppos[k];
        }
    }
    s->finite = 1;
    s->finite = allfin_f(s->L, 7 * NF) && (pa < 0 || allfin_d(s->P, 7 * NF));
    s->fexit = 0;
    s->open = 1;
    return 0;
}

/* gather the committed-local blocks of the subtree bones [g, j] (8 bones per pass over the frames; the state's other
 * bones are constant during a session, see above) and keep `finite` = every gathered input is finite.  A bone's
 * frame-level exit only depends on its own inputs and its ancestors' (earlier in the subtree order), so clearing
 * `finite` on a later block leaves the earlier bones' exits exact. */
static void sess_gather(ek_sess_t *s, int j)
{
    const int F = s->F, B = s->S.B, ns = s->nsub;
    const size_t NF = (size_t)F;
    while (s->g <= j) {
        const int j0 = s->g, j1 = j0 + 8 < ns ? j0 + 8 : ns;
        for (int f = 0; f < F; f++) {
            const float *row = s->S.cur_local + (size_t)f * B * 7;
            for (int jj = j0; jj < j1; jj++) memcpy(s->L + ((size_t)jj * NF + f) * 7, row + (size_t)s->T.sub[jj] * 7, 7 * sizeof(float));
        }
        if (s->finite && !allfin_f(s->L + 7 * NF * (size_t)j0, 7 * NF * (size_t)(j1 - j0))) s->finite = 0;
        s->g = j1;
    }
}

/* the frame-level early exit of ek_sess_eval (the caller vouches that the raw arrays are finite) */
void ek_sess_fexit(int on) { g_ss.fexit = on; }

void ek_sess_end(void) { g_ss.open = 0; }

/* after a commit on bone b (its local changed; the rest of the subtree and the parent rows did not): re-read bone b's
 * committed local into the session, which then stays valid for the next candidates of the same bone */
void ek_sess_refresh_b(void)
{
    ek_sess_t *s = &g_ss;
    const int F = s->F, B = s->S.B;
    const size_t NF = (size_t)F;
    for (int f = 0; f < F; f++) memcpy(s->L + 7 * (size_t)f, s->S.cur_local + ((size_t)f * B + s->T.sub[0]) * 7, 7 * sizeof(float));
    if (s->finite && !allfin_f(s->L, 7 * NF)) s->finite = 0;
}

/* the raw parent rows of bone 0 (for parent_raw candidates), gathered once per session from the bone-major AoS copies */
static void sess_raw_parent(const double *raw_rot_bm, const double *raw_pos_bm)
{
    ek_sess_t *s = &g_ss;
    const int F = s->F, pa = s->T.par_abs[0];
    const size_t NF = (size_t)F;
    if (s->have_raw || pa < 0) return;
    for (int f = 0; f < F; f++) {
        const double *prot = raw_rot_bm + ((size_t)pa * F + f) * 4, *ppos = raw_pos_bm + ((size_t)pa * F + f) * 3;
        for (int k = 0; k < 4; k++) s->PR[k * NF + f] = prot[k];
        for (int k = 0; k < 3; k++) s->PR[(4 + k) * NF + f] = ppos[k];
    }
    s->have_raw = 1;
}

/* Evaluate a candidate of the session's track: bone b's local from the track values v at the keys K [nK] (nK == F:
 * every frame) with integers quantize(v[K], step) + off (off NULL: none) -- or, step == NULL, the raw values rounded to
 * float32 (an unquantized track) -- then the subtree with the early exit of `mode` (0 none, 1 max > lim_max, 2 sum >
 * lim_sum or max > lim_max, 3 the growth offer's [esum, rem] pairs in lim_sum).  parent_raw: bone b chained onto its
 * parent's RAW object transform (raw_rot_bm / raw_pos_bm: bone-major AoS copies).
 * Returns the number of bones computed (nd == nsub: all passed their early-exit test), -1 on a NaN, -2 on allocation. */
int ek_sess_eval(const i64 *K, i64 nK, const i64 *off, const double *step, int parent_raw, const double *raw_rot_bm,
                 const double *raw_pos_bm, int mode, const double *lim_max, const double *lim_sum)
{
    ek_sess_t *s = &g_ss;
    if (!s->open) return -2;
    const int F = s->F, ns = s->nsub;
    const size_t NF = (size_t)F;
    const int c = s->cur;
    double *Rc = s->R[c], *Ec = s->E[c], *st = s->st[c];
    float *loc = s->loc[c];
    const ek_track *T = &s->T;
    /* bone b's local: the committed one with the track's components replaced */
    double *tmp = (double *)bget(&g_k[3], sizeof(double) * (7 * NF + 2 * (size_t)nK + 8));
    i64 *rows = (i64 *)bget(&g_k[2], sizeof(i64) * NF);
    i64 *nk = (i64 *)bget(&g_k[1], sizeof(i64) * (3 * (size_t)nK + 3));
    if (!tmp || !rows || !nk) return -2;
    double *vx = tmp, *sc = tmp + 3 * NF, *vk = sc + 4 * NF, *mm = vk + nK;
    if (step) {
        for (i64 k = 0; k < nK; k++) for (int cc = 0; cc < 3; cc++)
            nk[3 * k + cc] = q_int(T->v[3 * (size_t)K[k] + cc], step[cc]) + (off ? off[3 * k + cc] : 0);
        if (nK == F) {
            for (int cc = 0; cc < 3; cc++) for (i64 f = 0; f < F; f++) vx[cc * NF + f] = dq(nk[3 * f + cc], step[cc]);
        } else {
            for (i64 f = 0; f < F; f++) rows[f] = f;
            for (int cc = 0; cc < 3; cc++) interp_keys_col(nK, K, nk, cc, step[cc], F, rows, vk, mm, vx + cc * NF);
        }
    } else {
        for (int cc = 0; cc < 3; cc++) for (i64 f = 0; f < F; f++) vx[cc * NF + f] = (double)(float)T->v[3 * (size_t)f + cc];
    }
    /* loc (float32 [F, 7]) = committed local of b with the track's components (exp_map for a rotation) */
    memcpy(loc, s->L, sizeof(float) * 7 * NF);
    if (s->kind == 0) {
        double *theta = sc, *half = sc + NF, *sn = sc + 2 * NF, *cs = sc + 3 * NF;
        const double *x = vx, *y = vx + NF, *z = vx + 2 * NF;
        for (i64 r = 0; r < F; r++) {
            double q = 0.0;
            q += x[r] * x[r]; q += y[r] * y[r]; q += z[r] * z[r];
            theta[r] = sqrt(q); half[r] = 0.5 * theta[r];
        }
        ek_sincos(F, half, sn, cs);
        for (i64 r = 0; r < F; r++) {
            const double th = theta[r], den = th < 1e-30 ? 1e-30 : th;
            const double k = th > 1e-12 ? sn[r] / den : 0.5;
            float *o = loc + 7 * (size_t)r;
            o[0] = (float)(x[r] * k); o[1] = (float)(y[r] * k); o[2] = (float)(z[r] * k); o[3] = (float)cs[r];
        }
    } else {
        for (i64 r = 0; r < F; r++) { float *o = loc + 7 * (size_t)r; o[4] = (float)vx[r]; o[5] = (float)vx[NF + r]; o[6] = (float)vx[2 * NF + r]; }
    }
    if (parent_raw) sess_raw_parent(raw_rot_bm, raw_pos_bm);
    int fx = s->fexit && s->finite && mode >= 1;
    if (fx && !allfin_f(loc, 7 * NF)) fx = 0;
    if (fx && parent_raw && !allfin_d(s->PR, 7 * NF)) fx = 0;
    int ret = ns;
    for (int j = 0; j < ns; j++) {
        const int cb = T->sub[j], pa = T->par_abs[j], prel = T->par_rel[j];
        double *R = Rc + 7 * NF * j;
        double *q0 = R, *q1 = R + NF, *q2 = R + 2 * NF, *q3 = R + 3 * NF, *t0 = R + 4 * NF, *t1 = R + 5 * NF, *t2 = R + 6 * NF;
        if (j == 0) {
            for (int f = 0; f < F; f++) {
                const float *lf = loc + 7 * (size_t)f;
                q0[f] = (double)lf[0]; q1[f] = (double)lf[1]; q2[f] = (double)lf[2]; q3[f] = (double)lf[3];
                t0[f] = (double)lf[4]; t1[f] = (double)lf[5]; t2[f] = (double)lf[6];
            }
        } else {
            if (j >= s->g) sess_gather(s, j);
            if (!s->finite) fx = 0;
            const float *LL = s->L + 7 * NF * j;
            for (int f = 0; f < F; f++) {
                const float *lf = LL + 7 * (size_t)f;
                q0[f] = (double)lf[0]; q1[f] = (double)lf[1]; q2[f] = (double)lf[2]; q3[f] = (double)lf[3];
                t0[f] = (double)lf[4]; t1[f] = (double)lf[5]; t2[f] = (double)lf[6];
            }
        }
        const double *rc0 = s->S.raw_cols_s + (size_t)cb * 9 * NF, *rp0 = s->S.raw_pos_s + (size_t)cb * 3 * NF;
        const double *PP = 0;
        if (pa < 0) {
            st_norm(F, q0, q1, q2, q3);
        } else {
            if (j > 0 && prel >= 0) PP = Rc + 7 * NF * prel;
            else if (j == 0 && parent_raw) PP = s->PR;
            else if (j == 0) PP = s->P;
            else {                                        /* a parent outside this subtree prefix: committed state */
                double *X = (double *)bget(&g_k[0], sizeof(double) * 7 * NF);
                if (!X) return -2;
                for (int f = 0; f < F; f++) {
                    const double *prot = s->S.obj_rot + ((size_t)f * s->S.B + pa) * 4, *ppos = s->S.obj_pos + ((size_t)f * s->S.B + pa) * 3;
                    for (int k = 0; k < 4; k++) X[k * NF + f] = prot[k];
                    for (int k = 0; k < 3; k++) X[(4 + k) * NF + f] = ppos[k];
                }
                PP = X;
                if (fx && !allfin_d(X, 7 * NF)) fx = 0;
            }
            if (!fx) st_fk(F, q0, q1, q2, q3, t0, t1, t2, PP, PP + NF, PP + 2 * NF, PP + 3 * NF, PP + 4 * NF, PP + 5 * NF, PP + 6 * NF);
        }
        double *ej = Ec + NF * j;
        if (!fx) {
            st_err(F, q0, q1, q2, q3, t0, t1, t2, rc0, rc0 + NF, rc0 + 2 * NF, rc0 + 3 * NF, rc0 + 4 * NF, rc0 + 5 * NF, rc0 + 6 * NF,
                   rc0 + 7 * NF, rc0 + 8 * NF, rp0, rp0 + NF, rp0 + 2 * NF, s->S.shell, ej);
        } else {
            /* frame blocks, the max test after each (exact: max over a block > limit => the bone fails; inputs finite) */
            double bmx = 0.0;
            int failed = 0;
            for (int a = 0; a < F && !failed; a += 256) {
                const int m = F - a < 256 ? F - a : 256;
                if (PP) st_fk(m, q0 + a, q1 + a, q2 + a, q3 + a, t0 + a, t1 + a, t2 + a, PP + a, PP + NF + a, PP + 2 * NF + a, PP + 3 * NF + a,
                              PP + 4 * NF + a, PP + 5 * NF + a, PP + 6 * NF + a);
                st_err(m, q0 + a, q1 + a, q2 + a, q3 + a, t0 + a, t1 + a, t2 + a, rc0 + a, rc0 + NF + a, rc0 + 2 * NF + a, rc0 + 3 * NF + a,
                       rc0 + 4 * NF + a, rc0 + 5 * NF + a, rc0 + 6 * NF + a, rc0 + 7 * NF + a, rc0 + 8 * NF + a, rp0 + a, rp0 + NF + a,
                       rp0 + 2 * NF + a, s->S.shell, ej + a);
                for (int t = a; t < a + m; t++) if (t == 0 || ej[t] > bmx) bmx = ej[t];
                if (bmx > lim_max[j]) failed = 1;
            }
            if (failed) { st[3 * j] = bmx; st[3 * j + 1] = INFINITY; st[3 * j + 2] = 0.0; ret = j + 1; break; }
        }
        double mx = 0.0, ct = 0.0;
        if (err_stats(ej, F, s->S.p, &mx, &ct)) { ret = -1; break; }
        const double sm = np_sum(ej, F);
        st[3 * j] = mx; st[3 * j + 1] = sm; st[3 * j + 2] = ct;
        if (mode == 1 && F > 0 && mx > lim_max[j]) { ret = j + 1; break; }
        if (mode == 2 && F > 0 && (sm > lim_sum[j] || mx > lim_max[j])) { ret = j + 1; break; }
        if (mode == 3 && F > 0 && (mx > lim_max[j] || !(sm - lim_sum[2 * j] <= lim_sum[2 * j + 1] + 1e-12))) { ret = j + 1; break; }
    }
    s->nd[c] = ret;
    return ret;
}

/* commit the kept candidate straight into the search state (search.Search.commit_pending with the session's rows):
 * cur_local[:, b] = its local, obj_rot / obj_pos / err[:, c] = its rows, emax[c] = its max, cnt[c] = its count, for every
 * subtree bone c (a kept candidate was computed in full and holds no NaN, so its max is numpy's).  Frame blocks of 16:
 * the subtree's records of one frame are written together.  Returns 0, or -1 when there is no complete kept candidate. */
int ek_sess_commit(void)
{
    ek_sess_t *s = &g_ss;
    const int k = 1 - s->cur, nd = s->nd[k], F = s->F, B = s->S.B, ns = s->nsub, b = s->T.sub[0];
    const size_t NF = (size_t)F;
    if (!s->open || nd != ns) return -1;
    const float *loc = s->loc[k];
    for (int f = 0; f < F; f++) memcpy(s->S.cur_local + ((size_t)f * B + b) * 7, loc + 7 * (size_t)f, 7 * sizeof(float));
    for (int f0 = 0; f0 < F; f0 += 16) {
        const int f1 = f0 + 16 < F ? f0 + 16 : F;
        for (int j = 0; j < ns; j++) {
            const int c = s->T.sub[j];
            const double *R = s->R[k] + 7 * NF * j, *E = s->E[k] + NF * j;
            for (int f = f0; f < f1; f++) {
                double *o = s->S.obj_rot + ((size_t)f * B + c) * 4, *q = s->S.obj_pos + ((size_t)f * B + c) * 3;
                o[0] = R[f]; o[1] = R[NF + f]; o[2] = R[2 * NF + f]; o[3] = R[3 * NF + f];
                q[0] = R[4 * NF + f]; q[1] = R[5 * NF + f]; q[2] = R[6 * NF + f];
                s->S.err[(size_t)f * B + c] = E[f];
            }
        }
    }
    for (int j = 0; j < ns; j++) { const int c = s->T.sub[j]; s->S.emax[c] = s->st[k][3 * j]; s->S.cnt[c] = s->st[k][3 * j + 2]; }
    return 0;
}

/* search.Search.commit_pending's writes from the pending tuple: cur_local[:, b] = loc (float32 [F, 7]);
 * for each subtree bone c = sub[j]: obj_rot / obj_pos / err[:, c] = the arrays at addresses rp[j] / pp[j] / ep[j]
 * (C-contiguous float64 [F, 4] / [F, 3] / [F]), emax[c] = max(err_c), cnt[c] = cnt[j].  Frame blocks of 16 (the subtree's
 * records of one frame written together).  Returns 1 without writing anything when an err_c holds a NaN (numpy's max
 * propagates it: the caller keeps numpy's commit), 0 when done, -2 on allocation. */
static ek_buf g_cm;
int ek_commit_rows(int F, int B, int b, const float *loc, int ns, const int *sub, const i64 *rp, const i64 *pp, const i64 *ep,
                   const double *cnt, float *cur_local, double *obj_rot, double *obj_pos, double *err, double *emax, double *cnt_out)
{
    double *mx = (double *)bget(&g_cm, sizeof(double) * ((size_t)ns + 1));
    if (!mx) return -2;
    for (int j = 0; j < ns; j++) {
        const double *e = (const double *)(intptr_t)ep[j];
        double m = e[0];
        for (int f = 0; f < F; f++) {
            const double x = e[f];
            if (x != x) return 1;
            if (x > m) m = x;
        }
        mx[j] = m;
    }
    for (int f = 0; f < F; f++) memcpy(cur_local + ((size_t)f * B + b) * 7, loc + 7 * (size_t)f, 7 * sizeof(float));
    for (int f0 = 0; f0 < F; f0 += 16) {
        const int f1 = f0 + 16 < F ? f0 + 16 : F;
        for (int j = 0; j < ns; j++) {
            const int c = sub[j];
            const double *r = (const double *)(intptr_t)rp[j], *q = (const double *)(intptr_t)pp[j], *e = (const double *)(intptr_t)ep[j];
            for (int f = f0; f < f1; f++) {
                memcpy(obj_rot + ((size_t)f * B + c) * 4, r + 4 * (size_t)f, 4 * sizeof(double));
                memcpy(obj_pos + ((size_t)f * B + c) * 3, q + 3 * (size_t)f, 3 * sizeof(double));
                err[(size_t)f * B + c] = e[f];
            }
        }
    }
    for (int j = 0; j < ns; j++) { emax[sub[j]] = mx[j]; cnt_out[sub[j]] = cnt[j]; }
    return 0;
}

/* the per-bone statistics of the current candidate: out [3 nd] (max, sum, count per computed bone) */
void ek_sess_stats(double *out)
{
    const ek_sess_t *s = &g_ss;
    const int nd = s->nd[s->cur] > 0 ? s->nd[s->cur] : 0;
    memcpy(out, s->st[s->cur], sizeof(double) * 3 * (size_t)nd);
}

/* keep the current candidate (it passed: it becomes the pending one); the next evaluation reuses the other slot */
void ek_sess_keep(void) { g_ss.cur = 1 - g_ss.cur; }
