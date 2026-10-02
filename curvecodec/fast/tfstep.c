/* The fused integer transformer step of model.IntTf.step, in C.
 *
 * One call = one lockstep time step of the entropy model (embed -> L pre-LN blocks of sliding-window attention +
 * FFN -> final -> mixture head) for the ACTIVE curves only, with every curve's KV cache in C-owned memory. The result is
 * bit-identical to IntTf.step's rows for those curves (numpy computes every product through float64 BLAS, exact because
 * every partial sum stays below 2^53; here every product is an exact integer operation):
 *
 *   matmuls   int16 weights x int16 activations -> int32 sums (AVX-512 VNNI vpdpwssd / portable C), exact because a row is
 *             only sent through the int32 kernel when Cauchy-Schwarz proves sum_i |w_ji a_i| <= ||w_j|| ||a|| < 2^31
 *             (||w_j||^2 <= l2max2 precomputed per matrix, ||a||^2 computed per row). Rows that fail -- or hold values
 *             >= 2^15 (the Q12 residual stream into the final layer) -- are split into 2-4 limbs a = sum_k l_k 2^(b k)
 *             (low limbs in [0, 2^b), b chosen per matrix so that any such limb passes the bound), each limb exact in
 *             int32, recombined in int64. A row no limb split can handle is computed in scalar int64.
 *             (LayerNorm outputs have ||a|| ~ 2^15 by construction, so the LN-fed matmuls pass in one limb.)
 *   attention KV cache int16 (Q8 values; a curve whose k or v ever leaves int16 is PROMOTED to an int32 cache and the
 *             scalar int64 reference step for the rest of the clip). Scores q.k in int32 when ||q_h||^2 x the curve's
 *             running max ||k_h||^2 < 2^62 (else scalar int64); logits (s >> 8) + rel; the Q16 EXP_TAB (int32
 *             gather); p = (e << 16) // tot through a correctly-rounded double division (quotient < 2^17, exact floor:
 *             a non-integer quotient is >= 1/tot >= 2^-23 below the next integer, the rounding error < 2^-36);
 *             o = (sum p v) >> 16 with p split into p >> 8 (<= 256) and p & 255: both int32 sums exact because
 *             sum_w p_w <= 2^16 and |v| < 2^15.
 *   the rest  integer LayerNorm (Q16 rsqrt table, numpy's int64 wrap-around replicated with uint64), bias add, floor
 *             shifts, ReLU / clips: int64 scalar, exactly numpy's operations.
 * Only the active curves are computed: every row of IntTf.step is independent of the others, and a curve is active
 * exactly while t < n_c (the codec's lockstep), so an ended curve is never read again. The step checks that contract
 * (a curve must have been active at every previous step) and returns -1 otherwise.
 *
 * API (function-pointer friendly; fast/tfstep.py wraps it):
 *   void *tfs_model_create(const int64_t *P, const int64_t *const *A)   weights etc. (layout in tfs_model_create)
 *   void  tfs_model_free(void *model)
 *   void *tfs_state_create(void *model, int64_t C, int64_t pad)        caches of C curves = the state after `pad`
 *                                                                        zero-feature steps (IntTf.start(C, pad))
 *   void  tfs_state_free(void *state)
 *   int   tfs_set_threads(void *state, int nthreads, const int *cpus)   optional: share each step's curves among nthreads
 *         threads (curve c on thread c % nthreads, workers pinned to cpus[] if given); results identical
 *   int   tfs_step(void *state, int64_t t, int64_t n_act, const int64_t *act, const int64_t *x, int64_t *out)
 *         t = lockstep step 0.. (the state's own counter must equal t); act = n_act ascending curve ids; x = [n_act, NF]
 *         int64 Q8 features (row i = curve act[i]); out = [n_act, 3K] int64: logw[K] | mu[K] | logs[K] (logs clipped).
 *         Returns 0, -1 (contract violation), -2 (a value outside every exact path; never seen). After a nonzero return
 *         the state is undefined (re-run the clip on the numpy path).
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <stdatomic.h>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && !defined(TFS_PORTABLE)
#include <immintrin.h>
#define TFS_AVX512 1
#endif


typedef int64_t i64;
typedef int32_t i32;
typedef int16_t i16;
typedef uint64_t u64;

#define MAXL 32
#define RB 8                         /* curves per block (matmul weight reuse) */
#define MAXV (4 * RB)                /* virtual rows (limbs) per block */
#define KR 8                         /* rows per kernel call */
#define MAXH 64                      /* heads */
#define LIM62 ((u64)0x3fffffffffffffffULL)   /* 2^62 - 1 */

typedef struct {
    int N, M, Np, Mp;                /* inputs, outputs, padded inputs (even), padded outputs (multiple of 16) */
    i16 *Wp;                         /* packed [Np/2][Mp][2]: (W[j][2i], W[j][2i+1]) */
    i32 *Wr;                         /* raw [M][N] */
    i64 *b;                          /* [M] */
    u64 l2max2;                      /* max_j sum_i W[j][i]^2 */
    u64 thr;                         /* (2^62 - 1) / l2max2: a row with ||a||^2 <= thr is exact in int32 */
    int limb_bits;                   /* widest b <= 15 with N (2^b - 1)^2 <= thr */
} tfmat;

typedef struct {
    int D, NL, NH, dh, W, K, NF, FF, FD, log2d;
    int ws0, wsf, wsh, wsq[MAXL], wso[MAXL], ws1[MAXL], ws2[MAXL];
    i64 exp_lim, logs_lo, logs_hi, h_lim, ln_eps, rsq_bits, aq, hq;
    tfmat m0, mf, mh, mqkv[MAXL], mo[MAXL], m1[MAXL], m2[MAXL];
    i32 *rel;                        /* [NL][NH][W] */
    i32 *exp32;                      /* [exp_lim + 1] */
    i32 exp_nz;                      /* smallest index with exp32 != 0 (every z below it gives e = 0) */
    i64 *rsq; i64 n_rsq;
    int fast;                        /* the fast path's shape conditions hold */
    i64 curve_bytes;                 /* int16 cache bytes per curve */
    /* the pad template (one curve after `pad` zero steps) */
    i64 tpl_pad; int tpl_ok; int tpl_wide; i16 *tpl16; i32 *tplw; u64 tpl_kmax2[MAXL];
} tfmodel;

typedef struct {
    const tfmodel *m;
    i64 C, pad, t;
    i16 *c16;                        /* [C][NL][NH][2][dh * W] */
    i32 **wide;                      /* per curve NULL or [NL][2][W][D] */
    i64 *last;                       /* last lockstep step each curve was run at (-1 = none) */
    u64 *kmax2;                      /* [C][NL] running max ||k_h||^2 */
    /* scratch */
    i32 *relslot;                    /* [NL][NH][W] */
    i32 *h, *a, *o, *u, *f;          /* [RB][D] / [RB][max(D, FF, FD)] */
    i64 *acc;                        /* [RB][max(3D, FF, FD, 3K, D)] */
    i32 *qkv;                        /* [RB][3D] (q only is used after the cache write) */
    i16 *vr; i32 *S;                 /* limb rows [MAXV][Npmax], kernel outputs [4][Mpmax] */
    i32 *dots, *lg, *ex; i16 *phi, *plo; i16 *q16;
    i64 *xr;                         /* one row of int64 work (reference path) */
    i64 *refbuf;                     /* reference path scratch */
    int ldm;                         /* row stride of a / u / f / acc */
    int is_clone;                    /* a worker's view: shares everything but the scratch buffers */
    struct tfpool *pool;             /* worker threads (tfs_set_threads), NULL = single thread */
} tfstate;

#define TMAX 64
typedef struct tfpool {
    int nw;                          /* workers besides the caller */
    pthread_t th[TMAX];
    tfstate *sub[TMAX + 1];          /* sub[0] = the state itself, sub[k] = worker k's clone */
    _Atomic long gen;                /* bumped by the caller to start a step */
    _Atomic int done;                /* workers finished with the current step */
    _Atomic int quit;
    i64 T; const i64 *x; i64 *out;
    i64 *ids[TMAX + 1], *rows[TMAX + 1], n[TMAX + 1];
    int rc[TMAX + 1];
    int cpus[TMAX + 1];
} tfpool;

static inline i64 clampi(i64 v, i64 lo, i64 hi) { return v < lo ? lo : (v > hi ? hi : v); }

static inline int bitlen64(u64 v) { return v ? 64 - __builtin_clzll(v) : 0; }

/* ------------------------------------------------------------------------------------------------ matrices */
static int mat_init(tfmat *mt, const i64 *W, const i64 *b, int M, int N)
{
    mt->N = N; mt->M = M; mt->Np = (N + 1) & ~1; mt->Mp = (M + 15) & ~15;
    mt->Wp = (i16 *)calloc((size_t)mt->Np * mt->Mp, sizeof(i16));
    mt->Wr = (i32 *)malloc((size_t)M * N * sizeof(i32));
    mt->b = (i64 *)malloc((size_t)M * sizeof(i64));
    if (!mt->Wp || !mt->Wr || !mt->b) return -1;
    u64 l2 = 0;
    for (int j = 0; j < M; j++) {
        u64 s = 0;
        for (int i = 0; i < N; i++) {
            const i64 w = W[(size_t)j * N + i];
            if (w < -32767 || w > 32767) return -1;
            mt->Wr[(size_t)j * N + i] = (i32)w;
            mt->Wp[((size_t)(i >> 1) * mt->Mp + j) * 2 + (i & 1)] = (i16)w;
            s += (u64)(w * w);
        }
        if (s > l2) l2 = s;
        mt->b[j] = b[j];
    }
    mt->l2max2 = l2 ? l2 : 1;
    mt->thr = LIM62 / mt->l2max2;
    int bb = 0;
    for (int k = 1; k <= 15; k++) {
        const u64 lv = ((u64)1 << k) - 1;
        if ((u64)N * lv * lv <= mt->thr) bb = k;
    }
    mt->limb_bits = bb;
    return 0;
}

static void mat_free(tfmat *mt) { free(mt->Wp); free(mt->Wr); free(mt->b); }

/* S[r][0..Mp) = int32 sums of W . A[r] over nr <= 4 int16 rows (stride lda), exact by the caller's bound */
#ifdef TFS_AVX512
static inline __attribute__((always_inline)) void mm_tile(const tfmat *mt, int m0, int NR, int T, const i16 *A, int lda,
                                                          i32 *S, int lds)
{
    __m512i acc[8][4];
    for (int r = 0; r < NR; r++)
        for (int t = 0; t < T; t++) acc[r][t] = _mm512_setzero_si512();
    const int np2 = mt->Np >> 1, Mp = mt->Mp;
    for (int i2 = 0; i2 < np2; i2++) {
        const i16 *w = mt->Wp + ((size_t)i2 * Mp + m0) * 2;
        __m512i wv[4];
        for (int t = 0; t < T; t++) wv[t] = _mm512_loadu_si512((const void *)(w + 32 * t));
        for (int r = 0; r < NR; r++) {
            i32 pr; memcpy(&pr, A + (size_t)r * lda + 2 * i2, 4);
            const __m512i bc = _mm512_set1_epi32(pr);
            for (int t = 0; t < T; t++) acc[r][t] = _mm512_dpwssd_epi32(acc[r][t], bc, wv[t]);
        }
    }
    for (int r = 0; r < NR; r++)
        for (int t = 0; t < T; t++) _mm512_storeu_si512((void *)(S + (size_t)r * lds + m0 + 16 * t), acc[r][t]);
}

#define MM_CASE4(NR) \
    case NR: \
        for (; m0 + 64 <= Mp; m0 += 64) mm_tile(mt, m0, NR, 4, A, lda, S, lds); \
        for (; m0 < Mp; m0 += 16) mm_tile(mt, m0, NR, 1, A, lda, S, lds); \
        break;
#define MM_CASE2(NR) \
    case NR: \
        for (; m0 + 32 <= Mp; m0 += 32) mm_tile(mt, m0, NR, 2, A, lda, S, lds); \
        for (; m0 < Mp; m0 += 16) mm_tile(mt, m0, NR, 1, A, lda, S, lds); \
        break;

static void mm_i16(const tfmat *mt, int nr, const i16 *A, int lda, i32 *S, int lds)
{
    int m0 = 0; const int Mp = mt->Mp;
    switch (nr) { MM_CASE4(1) MM_CASE4(2) MM_CASE4(3) MM_CASE4(4) MM_CASE2(5) MM_CASE2(6) MM_CASE2(7) MM_CASE2(8) default: break; }
}
#else
static void mm_i16(const tfmat *mt, int nr, const i16 *A, int lda, i32 *S, int lds)
{
    const int np2 = mt->Np >> 1, Mp = mt->Mp;
    for (int r = 0; r < nr; r++) {
        i32 *s = S + (size_t)r * lds;
        const i16 *a = A + (size_t)r * lda;
        for (int j = 0; j < Mp; j++) s[j] = 0;
        for (int i2 = 0; i2 < np2; i2++) {
            const i32 a0 = a[2 * i2], a1 = a[2 * i2 + 1];
            const i16 *w = mt->Wp + (size_t)i2 * Mp * 2;
            for (int j = 0; j < Mp; j++) s[j] += a0 * (i32)w[2 * j] + a1 * (i32)w[2 * j + 1];
        }
    }
}
#endif

/* limb split of one int32 row for matrix mt -> nl int16 rows in L (stride Np); returns nl (1..4) and the limb width, or 0 */
static int make_limbs(const tfmat *mt, const i32 *a, i16 *L, int *bits)
{
    const int N = mt->N, Np = mt->Np;
    i32 amax = 0;
#ifdef TFS_AVX512
    /* one pass: max |a|, the int16 copy and sum a^2 (pairs of int16 squares via vpmaddwd: each < 2^31 when |a| < 2^15,
     * widened to int64 lanes before adding) */
    int i = 0;
    __m512i vmax = _mm512_setzero_si512(), vss = _mm512_setzero_si512();
    for (; i + 16 <= N; i += 16) {
        const __m512i v = _mm512_loadu_si512((const void *)(a + i));
        vmax = _mm512_max_epi32(vmax, _mm512_abs_epi32(v));
        const __m256i h = _mm512_cvtepi32_epi16(v);                        /* exact when |a| < 2^15 (checked below) */
        _mm256_storeu_si256((__m256i *)(L + i), h);
        const __m256i sq = _mm256_madd_epi16(h, h);                        /* 8 lanes: a[2k]^2 + a[2k+1]^2 */
        vss = _mm512_add_epi64(vss, _mm512_cvtepu32_epi64(sq));           /* each pair sum < 2^31: unsigned widening */
    }
    amax = _mm512_reduce_max_epi32(vmax);
    u64 ss = (u64)_mm512_reduce_add_epi64(vss);
    for (; i < N; i++) {
        const i32 v = a[i] < 0 ? -a[i] : a[i];
        if (v > amax) amax = v;
        L[i] = (i16)a[i];
        ss += (u64)((i64)a[i] * a[i]);
    }
    if (amax < 32768) {
        if (ss <= mt->thr) {
            for (int k = N; k < Np; k++) L[k] = 0;
            *bits = 0;
            return 1;
        }
    }
#else
    for (int i = 0; i < N; i++) { const i32 v = a[i] < 0 ? -a[i] : a[i]; if (v > amax) amax = v; }
    if (amax < 32768) {
        u64 ss = 0;
        for (int i = 0; i < N; i++) ss += (u64)((i64)a[i] * a[i]);
        if (ss <= mt->thr) {
            for (int i = 0; i < N; i++) L[i] = (i16)a[i];
            for (int i = N; i < Np; i++) L[i] = 0;
            *bits = 0;
            return 1;
        }
    }
#endif
    const int b = mt->limb_bits;
    if (b < 1) return 0;
    const i32 mask = (i32)((1u << b) - 1);
    for (int nl = 2; nl <= 4; nl++) {
        const int sh = b * (nl - 1);
        if (sh >= 31) break;
        u64 ss = 0; int ok = 1;
        for (int i = 0; i < N; i++) {
            const i32 tv = a[i] >> sh;
            if (tv < -32768 || tv > 32767) { ok = 0; break; }
            ss += (u64)((i64)tv * tv);
        }
        if (!ok || ss > mt->thr) continue;
        for (int k = 0; k < nl; k++) {
            i16 *Lk = L + (size_t)k * Np;
            for (int i = 0; i < N; i++) Lk[i] = (k == nl - 1) ? (i16)(a[i] >> sh) : (i16)((a[i] >> (b * k)) & mask);
            for (int i = N; i < Np; i++) Lk[i] = 0;
        }
        *bits = b;
        return nl;
    }
    return 0;
}

/* Y[r][j] = sum_i W[j][i] A[r][i] (exact int64, no bias) for R rows of int32 activations */
static void matvec(tfstate *s, const tfmat *mt, int R, const i32 *A, int lda, i64 *Y, int ldy)
{
    int nv = 0, vrow[MAXV], vsh[MAXV];
    const int Np = mt->Np, M = mt->M;
    for (int r = 0; r < R; r++) {
        int bits = 0;
        const int nl = make_limbs(mt, A + (size_t)r * lda, s->vr + (size_t)nv * Np, &bits);
        i64 *y = Y + (size_t)r * ldy;
        if (nl == 0) {                                  /* scalar int64 (never seen in practice) */
            const i32 *a = A + (size_t)r * lda;
            for (int j = 0; j < M; j++) {
                i64 acc = 0;
                const i32 *w = mt->Wr + (size_t)j * mt->N;
                for (int i = 0; i < mt->N; i++) acc += (i64)w[i] * a[i];
                y[j] = acc;
            }
            continue;
        }
        for (int k = 0; k < nl; k++) { vrow[nv + k] = r; vsh[nv + k] = k == 0 ? -1 : bits * k; }   /* -1: first limb (store) */
        nv += nl;
    }
    for (int v0 = 0; v0 < nv; v0 += KR) {
        const int nr = nv - v0 < KR ? nv - v0 : KR;
        mm_i16(mt, nr, s->vr + (size_t)v0 * Np, Np, s->S, mt->Mp);
        for (int k = 0; k < nr; k++) {
            i64 *restrict y = Y + (size_t)vrow[v0 + k] * ldy;
            const i32 *restrict S = s->S + (size_t)k * mt->Mp;
            const int sh = vsh[v0 + k];
            if (sh < 0) for (int j = 0; j < M; j++) y[j] = (i64)S[j];
            else for (int j = 0; j < M; j++) y[j] += (i64)((u64)(i64)S[j] << sh);
        }
    }
}

/* post-ops on the int64 matmul rows (numpy: bias add, floor shift, clips) */
static void post_clamp(const i64 *restrict Y, const i64 *restrict b, int M, int sh, i64 lo, i64 hi, int shl, i32 *restrict out)
{
    int j = 0;
#ifdef TFS_AVX512
    const __m128i c = _mm_cvtsi32_si128(sh);
    const __m512i vlo = _mm512_set1_epi64(lo), vhi = _mm512_set1_epi64(hi);
    const __m128i cl = _mm_cvtsi32_si128(shl);
    for (; j + 8 <= M; j += 8) {
        __m512i v = _mm512_sra_epi64(_mm512_add_epi64(_mm512_loadu_si512((const void *)(Y + j)), _mm512_loadu_si512((const void *)(b + j))), c);
        v = _mm512_sll_epi64(_mm512_min_epi64(_mm512_max_epi64(v, vlo), vhi), cl);
        _mm256_storeu_si256((__m256i *)(out + j), _mm512_cvtepi64_epi32(v));
    }
#endif
    for (; j < M; j++) out[j] = (i32)(clampi((Y[j] + b[j]) >> sh, lo, hi) << shl);
}

static void post_resid(const i64 *restrict Y, const i64 *restrict b, int M, int sh, i64 H, i32 *restrict h)
{
    int j = 0;
#ifdef TFS_AVX512
    const __m128i c = _mm_cvtsi32_si128(sh);
    const __m512i vlo = _mm512_set1_epi64(-H), vhi = _mm512_set1_epi64(H);
    for (; j + 8 <= M; j += 8) {
        __m512i v = _mm512_sra_epi64(_mm512_add_epi64(_mm512_loadu_si512((const void *)(Y + j)), _mm512_loadu_si512((const void *)(b + j))), c);
        v = _mm512_add_epi64(v, _mm512_cvtepi32_epi64(_mm256_loadu_si256((const __m256i *)(h + j))));
        v = _mm512_min_epi64(_mm512_max_epi64(v, vlo), vhi);
        _mm256_storeu_si256((__m256i *)(h + j), _mm512_cvtepi64_epi32(v));
    }
#endif
    for (; j < M; j++) h[j] = (i32)clampi((i64)h[j] + ((Y[j] + b[j]) >> sh), -H, H);
}

/* out = (Y + b) >> sh (no clip), with the min / max of the results (range checks) */
static void post_range(const i64 *restrict Y, const i64 *restrict b, int M, int sh, i32 *restrict out, i64 *mn, i64 *mx)
{
    int j = 0;
    i64 lo = INT64_MAX, hi = INT64_MIN;
#ifdef TFS_AVX512
    const __m128i c = _mm_cvtsi32_si128(sh);
    __m512i vmn = _mm512_set1_epi64(INT64_MAX), vmx = _mm512_set1_epi64(INT64_MIN);
    for (; j + 8 <= M; j += 8) {
        const __m512i v = _mm512_sra_epi64(_mm512_add_epi64(_mm512_loadu_si512((const void *)(Y + j)), _mm512_loadu_si512((const void *)(b + j))), c);
        vmn = _mm512_min_epi64(vmn, v); vmx = _mm512_max_epi64(vmx, v);
        _mm256_storeu_si256((__m256i *)(out + j), _mm512_cvtepi64_epi32(v));
    }
    lo = _mm512_reduce_min_epi64(vmn); hi = _mm512_reduce_max_epi64(vmx);
#endif
    for (; j < M; j++) {
        const i64 v = (Y[j] + b[j]) >> sh;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        out[j] = (i32)v;
    }
    *mn = lo; *mx = hi;
}

/* ------------------------------------------------------------------------------------------------ LayerNorm */
static inline i64 inv_sqrt_q16_of_q24(const tfmodel *m, i64 v)
{
    if (v < 1) v = 1;
    const i64 e = bitlen64((u64)v) - 1;
    const i64 e2 = e & ~(i64)1;
    const i64 sh = e2 - m->rsq_bits;
    const i64 mm = sh >= 0 ? (v >> sh) : (i64)((u64)v << (-sh));
    const i64 r = m->rsq[mm - ((i64)1 << m->rsq_bits)];
    const i64 k = 12 - (e2 >> 1);
    return k >= 0 ? (i64)((u64)r << k) : (r >> (-k));
}

/* numpy: mean = sum(h) >> log2d; c = h - mean; var = (sum(c*c) >> log2d) + eps (int64, wraps); a = (c * inv) >> 16 */
static void ln_row(const tfmodel *m, const i32 *h, i32 *a)
{
    const int D = m->D;
    i64 s = 0;
    for (int i = 0; i < D; i++) s += h[i];
    const i64 mean = s >> m->log2d;
    u64 v = 0;
    for (int i = 0; i < D; i++) { const i64 c = (i64)h[i] - mean; v += (u64)(c * c); }
    const i64 var = ((i64)v >> m->log2d) + m->ln_eps;
    const i64 inv = inv_sqrt_q16_of_q24(m, var);
    for (int i = 0; i < D; i++) a[i] = (i32)((((i64)h[i] - mean) * inv) >> 16);
}

static void ln_row64(const tfmodel *m, const i64 *h, i64 *a)
{
    const int D = m->D;
    i64 s = 0;
    for (int i = 0; i < D; i++) s += h[i];
    const i64 mean = s >> m->log2d;
    u64 v = 0;
    for (int i = 0; i < D; i++) { const i64 c = h[i] - mean; v += (u64)c * (u64)c; }
    const i64 var = ((i64)v >> m->log2d) + m->ln_eps;
    const i64 inv = inv_sqrt_q16_of_q24(m, var);
    for (int i = 0; i < D; i++) a[i] = (i64)((u64)(h[i] - mean) * (u64)inv) >> 16;
}

/* ------------------------------------------------------------------------------------------------ model */
/* P (int64): [0] D [1] NL [2] NH [3] W [4] K [5] NF [6] FF [7] FD [8] ws0 [9] wsf [10] wsh [11] exp_lim [12] logs_lo
 * [13] logs_hi [14] h_lim [15] ln_eps [16] rsq_bits [17] n_rsq [18] aq [19] hq, then per layer l at 20 + 4 l: wsq wso ws1 ws2.
 * A (int64 arrays): [0] W0 [1] b0 [2] Wf [3] bf [4] Wh [5] bh [6] EXP_TAB [7] RSQRT_TAB, then per layer l at 8 + 9 l:
 * Wqkv bqkv Wo bo W1 b1 W2 b2 rel.  Weights row-major [out][in] as IntTf holds them. */
void tfs_model_free(void *vm);

void *tfs_model_create(const i64 *P, const i64 *const *A)
{
    tfmodel *m = (tfmodel *)calloc(1, sizeof(tfmodel));
    if (!m) return NULL;
    m->D = (int)P[0]; m->NL = (int)P[1]; m->NH = (int)P[2]; m->W = (int)P[3]; m->K = (int)P[4]; m->NF = (int)P[5];
    m->FF = (int)P[6]; m->FD = (int)P[7]; m->ws0 = (int)P[8]; m->wsf = (int)P[9]; m->wsh = (int)P[10];
    m->exp_lim = P[11]; m->logs_lo = P[12]; m->logs_hi = P[13]; m->h_lim = P[14]; m->ln_eps = P[15]; m->rsq_bits = P[16];
    m->n_rsq = P[17]; m->aq = P[18]; m->hq = P[19];
    if (m->NL < 1 || m->NL > MAXL || m->NH < 1 || m->NH > MAXH || m->D % m->NH) { free(m); return NULL; }
    m->dh = m->D / m->NH;
    m->log2d = 0; while ((1 << m->log2d) < m->D) m->log2d++;
    if ((1 << m->log2d) != m->D) { free(m); return NULL; }
    const int D = m->D, NL = m->NL;
    for (int l = 0; l < NL; l++) {
        m->wsq[l] = (int)P[20 + 4 * l]; m->wso[l] = (int)P[21 + 4 * l]; m->ws1[l] = (int)P[22 + 4 * l]; m->ws2[l] = (int)P[23 + 4 * l];
    }
    int bad = 0;
    bad |= mat_init(&m->m0, A[0], A[1], D, m->NF);
    bad |= mat_init(&m->mf, A[2], A[3], m->FD, D);
    bad |= mat_init(&m->mh, A[4], A[5], 3 * m->K, m->FD);
    for (int l = 0; l < NL; l++) {
        const i64 *const *L = A + 8 + 9 * l;
        bad |= mat_init(&m->mqkv[l], L[0], L[1], 3 * D, D);
        bad |= mat_init(&m->mo[l], L[2], L[3], D, D);
        bad |= mat_init(&m->m1[l], L[4], L[5], m->FF, D);
        bad |= mat_init(&m->m2[l], L[6], L[7], D, m->FF);
    }
    m->rel = (i32 *)malloc((size_t)NL * m->NH * m->W * sizeof(i32));
    m->exp32 = (i32 *)malloc((size_t)(m->exp_lim + 1) * sizeof(i32));
    m->rsq = (i64 *)malloc((size_t)m->n_rsq * sizeof(i64));
    if (bad || !m->rel || !m->exp32 || !m->rsq) { tfs_model_free(m); return NULL; }
    for (int l = 0; l < NL; l++)
        for (int k = 0; k < m->NH * m->W; k++) m->rel[(size_t)l * m->NH * m->W + k] = (i32)A[8 + 9 * l + 8][k];
    for (i64 k = 0; k <= m->exp_lim; k++) m->exp32[k] = (i32)A[6][k];
    m->exp_nz = (i32)m->exp_lim;
    for (i64 k = m->exp_lim; k >= 0; k--) if (m->exp32[k] != 0) m->exp_nz = (i32)k;
    for (i64 k = m->exp_nz; k <= m->exp_lim; k++) if (m->exp32[k] == 0) m->exp_nz = 0;   /* zeros only below exp_nz, else no skipping */
    for (i64 k = 0; k < m->n_rsq; k++) m->rsq[k] = A[7][k];
    /* fast-path shape conditions (else every curve runs the scalar reference step) */
    m->fast = (m->dh % 16 == 0) && (m->W % 16 == 0) && (m->W <= 2048) && (m->exp32[0] == 0) && (m->FD == D) && (m->exp_lim < (1 << 20));
    for (int k = 0; k <= m->exp_lim; k++) if (A[6][k] < 0 || A[6][k] > 65536) m->fast = 0;
    for (int l = 0; l < NL; l++) for (int k = 0; k < m->NH * m->W; k++) {
        const i64 r = A[8 + 9 * l + 8][k]; if (r < -(1 << 20) || r > (1 << 20)) m->fast = 0;
    }
    m->curve_bytes = (i64)NL * m->NH * 2 * m->dh * m->W * (i64)sizeof(i16);
    m->tpl_pad = -1;
    return m;
}

void tfs_model_free(void *vm)
{
    tfmodel *m = (tfmodel *)vm;
    if (!m) return;
    mat_free(&m->m0); mat_free(&m->mf); mat_free(&m->mh);
    for (int l = 0; l < m->NL; l++) { mat_free(&m->mqkv[l]); mat_free(&m->mo[l]); mat_free(&m->m1[l]); mat_free(&m->m2[l]); }
    free(m->rel); free(m->exp32); free(m->rsq); free(m->tpl16); free(m->tplw);
    free(m);
}

/* ------------------------------------------------------------------------------------------------ state */
static inline i16 *kblk(const tfstate *s, i64 c, int l, int h)
{
    const tfmodel *m = s->m;
    return s->c16 + (((size_t)c * m->NL + l) * m->NH + h) * 2 * (size_t)m->dh * m->W;
}

static i64 wide_len(const tfmodel *m) { return (i64)m->NL * 2 * m->W * m->D; }

void tfs_state_free(void *vs);
static int run_pad_template(tfmodel *m, i64 pad);

static int alloc_scratch(tfstate *s)
{
    const tfmodel *m = s->m;
    const int D = m->D, W = m->W;
    int mx = 3 * D; if (m->FF > mx) mx = m->FF; if (m->FD > mx) mx = m->FD; if (3 * m->K > mx) mx = 3 * m->K;
    if (m->NF > mx) mx = m->NF;
    s->ldm = mx;
    int npmax = m->m0.Np, mpmax = 0;
    const tfmat *all[4 + 4 * MAXL]; int na = 0;
    all[na++] = &m->m0; all[na++] = &m->mf; all[na++] = &m->mh;
    for (int l = 0; l < m->NL; l++) { all[na++] = &m->mqkv[l]; all[na++] = &m->mo[l]; all[na++] = &m->m1[l]; all[na++] = &m->m2[l]; }
    for (int k = 0; k < na; k++) { if (all[k]->Np > npmax) npmax = all[k]->Np; if (all[k]->Mp > mpmax) mpmax = all[k]->Mp; }
    s->h = (i32 *)calloc((size_t)RB * D, sizeof(i32));
    s->a = (i32 *)calloc((size_t)RB * mx, sizeof(i32));
    s->o = (i32 *)calloc((size_t)RB * D, sizeof(i32));
    s->u = (i32 *)calloc((size_t)RB * mx, sizeof(i32));
    s->f = (i32 *)calloc((size_t)RB * mx, sizeof(i32));
    s->acc = (i64 *)calloc((size_t)RB * mx, sizeof(i64));
    s->qkv = (i32 *)calloc((size_t)RB * 3 * D, sizeof(i32));
    s->vr = (i16 *)calloc((size_t)MAXV * npmax, sizeof(i16));
    s->S = (i32 *)calloc((size_t)KR * mpmax, sizeof(i32));
    s->dots = (i32 *)calloc((size_t)m->NH * W + 16, sizeof(i32));
    s->lg = (i32 *)calloc((size_t)m->NH * W + 16, sizeof(i32));
    s->ex = (i32 *)calloc((size_t)m->NH * W + 16, sizeof(i32));
    s->phi = (i16 *)calloc((size_t)m->NH * W + 32, sizeof(i16));
    s->plo = (i16 *)calloc((size_t)m->NH * W + 32, sizeof(i16));
    s->q16 = (i16 *)calloc((size_t)m->D + 32, sizeof(i16));
    s->xr = (i64 *)calloc((size_t)(m->NF > mx ? m->NF : mx) + 16, sizeof(i64));
    s->refbuf = (i64 *)calloc((size_t)8 * mx + 4 * W + 64, sizeof(i64));
    return (s->h && s->a && s->o && s->u && s->f && s->acc && s->qkv && s->vr && s->S && s->dots && s->lg && s->ex && s->phi &&
            s->plo && s->q16 && s->xr && s->refbuf) ? 0 : -1;
}

static void free_scratch(tfstate *s)
{
    free(s->h); free(s->a); free(s->o); free(s->u); free(s->f); free(s->acc); free(s->qkv); free(s->vr); free(s->S);
    free(s->dots); free(s->lg); free(s->ex); free(s->phi); free(s->plo); free(s->q16); free(s->xr); free(s->refbuf);
}

static tfstate *state_alloc(const tfmodel *m, i64 C)
{
    tfstate *s = (tfstate *)calloc(1, sizeof(tfstate));
    if (!s) return NULL;
    s->m = m; s->C = C;
    const int W = m->W;
    s->c16 = (i16 *)calloc((size_t)C * (m->curve_bytes / sizeof(i16)) + 32, sizeof(i16));
    s->wide = (i32 **)calloc((size_t)C, sizeof(i32 *));
    s->last = (i64 *)malloc((size_t)C * sizeof(i64));
    s->kmax2 = (u64 *)calloc((size_t)C * m->NL, sizeof(u64));
    s->relslot = (i32 *)malloc((size_t)m->NL * m->NH * W * sizeof(i32));
    if (alloc_scratch(s) || !s->c16 || !s->wide || !s->last || !s->kmax2 || !s->relslot) {
        tfs_state_free(s); return NULL;
    }
    for (i64 c = 0; c < C; c++) s->last[c] = -1;
    return s;
}

static void pool_stop(tfstate *s);

void tfs_state_free(void *vs)
{
    tfstate *s = (tfstate *)vs;
    if (!s) return;
    pool_stop(s);
    if (!s->is_clone) {
        if (s->wide) for (i64 c = 0; c < s->C; c++) free(s->wide[c]);
        free(s->c16); free(s->wide); free(s->last); free(s->kmax2); free(s->relslot);
    }
    free_scratch(s);
    free(s);
}

/* int16 cache of curve c -> int32 wide cache [NL][2][W][D] (slot-major, as the reference step reads it) */
static int promote(tfstate *s, i64 c)
{
    const tfmodel *m = s->m;
    if (s->wide[c]) return 0;
    const int D = m->D, W = m->W, dh = m->dh;
    i32 *w = (i32 *)malloc((size_t)wide_len(m) * sizeof(i32));
    if (!w) return -2;
    for (int l = 0; l < m->NL; l++)
        for (int h = 0; h < m->NH; h++) {
            const i16 *Kb = kblk(s, c, l, h), *Vb = Kb + (size_t)dh * W;
            i32 *Kw = w + ((size_t)l * 2 + 0) * W * D, *Vw = w + ((size_t)l * 2 + 1) * W * D;
            for (int sl = 0; sl < W; sl++)
                for (int i = 0; i < dh; i++) {
                    Kw[(size_t)sl * D + h * dh + i] = Kb[((size_t)(i >> 1) * W + sl) * 2 + (i & 1)];
                    Vw[(size_t)sl * D + h * dh + i] = Vb[((size_t)(sl >> 1) * dh + i) * 2 + (sl & 1)];
                }
        }
    s->wide[c] = w;
    return 0;
}

/* ------------------------------------------------------------------------------------------------ reference step */
/* The scalar int64 step of one curve on its wide cache (T = model time incl. pad): numpy's operations one by one. */
static int ref_step(tfstate *s, i64 c, i64 T, const i64 *x, i64 *out)
{
    const tfmodel *m = s->m;
    const int D = m->D, W = m->W, dh = m->dh, NH = m->NH, K = m->K;
    const int slot = (int)(T % W);
    i32 *cw = s->wide[c];
    i64 *h = s->refbuf, *a = h + D, *acc = a + (m->FF > D ? m->FF : D) + 8, *u = acc + 3 * D + m->FF + m->FD + 3 * K + 8;
    i64 *lgw = u + m->FF + m->FD + 8;
    /* embed */
    for (int j = 0; j < D; j++) {
        i64 sacc = 0;
        const i32 *w = m->m0.Wr + (size_t)j * m->NF;
        for (int i = 0; i < m->NF; i++) sacc += (i64)w[i] * x[i];
        i64 e = (sacc + m->m0.b[j]) >> (m->ws0 - m->aq);
        if (e < 0) e = 0;
        if (e > ((i64)1 << 24)) e = (i64)1 << 24;
        h[j] = e << (m->hq - m->aq);
    }
    for (int l = 0; l < m->NL; l++) {
        const tfmat *mq = &m->mqkv[l];
        i32 *Kw = cw + ((size_t)l * 2 + 0) * W * D, *Vw = cw + ((size_t)l * 2 + 1) * W * D;
        ln_row64(m, h, a);
        for (int j = 0; j < 3 * D; j++) {
            i64 sacc = 0;
            const i32 *w = mq->Wr + (size_t)j * D;
            for (int i = 0; i < D; i++) sacc += (i64)w[i] * a[i];
            acc[j] = (sacc + mq->b[j]) >> (m->wsq[l] + m->hq - m->aq);
        }
        for (int j = 0; j < D; j++) {
            if (acc[D + j] < INT32_MIN || acc[D + j] > INT32_MAX || acc[2 * D + j] < INT32_MIN || acc[2 * D + j] > INT32_MAX) return -2;
            Kw[(size_t)slot * D + j] = (i32)acc[D + j]; Vw[(size_t)slot * D + j] = (i32)acc[2 * D + j];
        }
        for (int hh = 0; hh < NH; hh++) {
            const i64 *q = acc + hh * dh;
            const i32 *rel = m->rel + ((size_t)l * NH + hh) * W;
            i64 mx = INT64_MIN;
            for (int w = 0; w < W; w++) {
                const i64 off = ((T - w) % W + W) % W;
                i64 v;
                if (off <= T) {
                    i64 d = 0;
                    for (int i = 0; i < dh; i++) d += q[i] * (i64)Kw[(size_t)w * D + hh * dh + i];
                    v = (d >> 8) + rel[off];
                } else v = -((i64)1 << 40);
                lgw[w] = v; if (v > mx) mx = v;
            }
            i64 tot = 0;
            for (int w = 0; w < W; w++) {
                const i64 off = ((T - w) % W + W) % W;
                const i64 e = off <= T ? m->exp32[clampi(lgw[w] - mx, -m->exp_lim, 0) + m->exp_lim] : 0;
                lgw[w] = e; tot += e;
            }
            for (int i = 0; i < dh; i++) {
                i64 sacc = 0;
                for (int w = 0; w < W; w++) sacc += ((lgw[w] << 16) / tot) * (i64)Vw[(size_t)w * D + hh * dh + i];
                u[hh * dh + i] = sacc >> 16;
            }
        }
        const tfmat *mo = &m->mo[l];
        for (int j = 0; j < D; j++) {
            i64 sacc = 0;
            const i32 *w = mo->Wr + (size_t)j * D;
            for (int i = 0; i < D; i++) sacc += (i64)w[i] * u[i];
            h[j] = clampi(h[j] + ((sacc + mo->b[j]) >> (m->wso[l] - (m->hq - m->aq))), -m->h_lim, m->h_lim);
        }
        ln_row64(m, h, a);
        const tfmat *m1 = &m->m1[l], *m2 = &m->m2[l];
        for (int j = 0; j < m->FF; j++) {
            i64 sacc = 0;
            const i32 *w = m1->Wr + (size_t)j * D;
            for (int i = 0; i < D; i++) sacc += (i64)w[i] * a[i];
            u[j] = clampi((sacc + m1->b[j]) >> (m->ws1[l] + m->hq - m->aq), 0, (i64)1 << 24);
        }
        for (int j = 0; j < D; j++) {
            i64 sacc = 0;
            const i32 *w = m2->Wr + (size_t)j * m->FF;
            for (int i = 0; i < m->FF; i++) sacc += (i64)w[i] * u[i];
            acc[j] = sacc;
        }
        for (int j = 0; j < D; j++)
            h[j] = clampi(h[j] + ((acc[j] + m2->b[j]) >> (m->ws2[l] - (m->hq - m->aq))), -m->h_lim, m->h_lim);
    }
    for (int j = 0; j < m->FD; j++) {
        i64 sacc = 0;
        const i32 *w = m->mf.Wr + (size_t)j * D;
        for (int i = 0; i < D; i++) sacc += (i64)w[i] * h[i];
        u[j] = clampi((sacc + m->mf.b[j]) >> (m->wsf + m->hq - m->aq), 0, (i64)1 << 24);
    }
    for (int j = 0; j < 3 * K; j++) {
        i64 sacc = 0;
        const i32 *w = m->mh.Wr + (size_t)j * m->FD;
        for (int i = 0; i < m->FD; i++) sacc += (i64)w[i] * u[i];
        const i64 o = (sacc + m->mh.b[j]) >> m->wsh;
        out[j] = j >= 2 * K ? clampi(o, m->logs_lo, m->logs_hi) : o;
    }
    return 0;
}

/* ------------------------------------------------------------------------------------------------ fast attention */
/* dots[w] = q . K[w] for one head (int32 exact by the caller's bound) */
static void head_scores(const tfmodel *m, const i16 *Kb, const i16 *q16, i32 *dots)
{
    const int W = m->W, dh = m->dh;
#ifdef TFS_AVX512
    int b0 = 0;
    for (; b0 + 128 <= W; b0 += 128) {
        __m512i acc[8];
        for (int bb = 0; bb < 8; bb++) acc[bb] = _mm512_setzero_si512();
        for (int i2 = 0; i2 < dh / 2; i2++) {
            i32 pr; memcpy(&pr, q16 + 2 * i2, 4);
            const __m512i bc = _mm512_set1_epi32(pr);
            const i16 *kr = Kb + ((size_t)i2 * W + b0) * 2;
            for (int bb = 0; bb < 8; bb++) acc[bb] = _mm512_dpwssd_epi32(acc[bb], bc, _mm512_loadu_si512((const void *)(kr + 32 * bb)));
        }
        for (int bb = 0; bb < 8; bb++) _mm512_storeu_si512((void *)(dots + b0 + 16 * bb), acc[bb]);
    }
    for (; b0 < W; b0 += 16) {
        __m512i acc = _mm512_setzero_si512();
        for (int i2 = 0; i2 < dh / 2; i2++) {
            i32 pr; memcpy(&pr, q16 + 2 * i2, 4);
            acc = _mm512_dpwssd_epi32(acc, _mm512_set1_epi32(pr), _mm512_loadu_si512((const void *)(Kb + ((size_t)i2 * W + b0) * 2)));
        }
        _mm512_storeu_si512((void *)(dots + b0), acc);
    }
#else
    for (int w = 0; w < W; w++) dots[w] = 0;
    for (int i2 = 0; i2 < dh / 2; i2++) {
        const i32 q0 = q16[2 * i2], q1 = q16[2 * i2 + 1];
        const i16 *kr = Kb + (size_t)i2 * W * 2;
        for (int w = 0; w < W; w++) dots[w] += q0 * (i32)kr[2 * w] + q1 * (i32)kr[2 * w + 1];
    }
#endif
}

/* The softmax of one head in three phases (called for all heads of a curve phase by phase, so the heads' dependency
 * chains -- reductions, gathers -- overlap):
 *   sm_logits  lg = (dots >> 8) + relslot, returns max lg
 *   sm_exp     e = EXP_TAB[max(lg - max, -EXP_LIM) + EXP_LIM] (int32 gather), returns tot = sum e (< 2^31: W <= 2^15)
 *   sm_p       p = (e << 16) // tot. Returns 1 when every p <= 32767 (then phi = p: one PV pass), else 0 with
 *              phi = p >> 8, plo = p & 255 (two passes). The quotient: q0 = trunc(float(e) * float(65536 / tot)) is within
 *              1 of floor (two float32 roundings, relative error < 2^-22, quotient <= 2^16), corrected with the exact
 *              remainder r = (e << 16) - q0 tot computed modulo 2^32 (|r| < 2 tot < 2^24: the wrapped int32 is exact). */
static inline i32 sm_logits(const tfmodel *m, const i32 *dots, const i32 *relslot, i32 *lg)
{
    const int W = m->W;
#ifdef TFS_AVX512
    __m512i vmx = _mm512_set1_epi32(INT32_MIN);
    for (int w = 0; w < W; w += 16) {
        const __m512i v = _mm512_add_epi32(_mm512_srai_epi32(_mm512_loadu_si512((const void *)(dots + w)), 8),
                                           _mm512_loadu_si512((const void *)(relslot + w)));
        _mm512_storeu_si512((void *)(lg + w), v);
        vmx = _mm512_max_epi32(vmx, v);
    }
    return _mm512_reduce_max_epi32(vmx);
#else
    i32 mx = INT32_MIN;
    for (int w = 0; w < W; w++) { const i32 v = (dots[w] >> 8) + relslot[w]; lg[w] = v; if (v > mx) mx = v; }
    return mx;
#endif
}

static inline i32 sm_exp(const tfmodel *m, const i32 *lg, i32 mx, i32 *ex)
{
    const int W = m->W;
    const i32 el = (i32)m->exp_lim;
#ifdef TFS_AVX512
    const __m512i vm = _mm512_set1_epi32(mx), vlo = _mm512_set1_epi32(-el), voff = _mm512_set1_epi32(el);
    __m512i vt = _mm512_setzero_si512();
    for (int w = 0; w < W; w += 16) {
        const __m512i z = _mm512_max_epi32(_mm512_sub_epi32(_mm512_loadu_si512((const void *)(lg + w)), vm), vlo);
        const __m512i e = _mm512_i32gather_epi32(_mm512_add_epi32(z, voff), (const void *)m->exp32, 4);
        _mm512_storeu_si512((void *)(ex + w), e);
        vt = _mm512_add_epi32(vt, e);
    }
    return _mm512_reduce_add_epi32(vt);
#else
    i32 tot = 0;
    for (int w = 0; w < W; w++) {
        i32 z = lg[w] - mx; if (z < -el) z = -el;
        ex[w] = m->exp32[z + el]; tot += ex[w];
    }
    return tot;
#endif
}

static inline int sm_p(const tfmodel *m, const i32 *ex, i32 tot, i32 *pb, i16 *phi, i16 *plo)
{
    const int W = m->W;
#ifdef TFS_AVX512
    const __m512 vc = _mm512_set1_ps((float)(65536.0 / (double)tot));
    const __m512i vtot = _mm512_set1_epi32(tot), one = _mm512_set1_epi32(1), zero = _mm512_setzero_si512();
    __m512i pmx = zero;
    for (int w = 0; w < W; w += 16) {
        const __m512i e = _mm512_loadu_si512((const void *)(ex + w));
        __m512i q = _mm512_cvttps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(e), vc));
        const __m512i r = _mm512_sub_epi32(_mm512_slli_epi32(e, 16), _mm512_mullo_epi32(q, vtot));
        q = _mm512_mask_sub_epi32(q, _mm512_cmplt_epi32_mask(r, zero), q, one);
        q = _mm512_mask_add_epi32(q, _mm512_cmpge_epi32_mask(r, vtot), q, one);
        pmx = _mm512_max_epi32(pmx, q);
        _mm512_storeu_si512((void *)(pb + w), q);
    }
    if (_mm512_reduce_max_epi32(pmx) <= 32767) {
        for (int w = 0; w < W; w += 16)
            _mm256_storeu_si256((__m256i *)(phi + w), _mm512_cvtepi32_epi16(_mm512_loadu_si512((const void *)(pb + w))));
        return 1;
    }
    const __m512i m255 = _mm512_set1_epi32(255);
    for (int w = 0; w < W; w += 16) {
        const __m512i p = _mm512_loadu_si512((const void *)(pb + w));
        _mm256_storeu_si256((__m256i *)(phi + w), _mm512_cvtepi32_epi16(_mm512_srli_epi32(p, 8)));
        _mm256_storeu_si256((__m256i *)(plo + w), _mm512_cvtepi32_epi16(_mm512_and_si512(p, m255)));
    }
    return 0;
#else
    i64 pm = 0;
    for (int w = 0; w < W; w++) { const i64 p = ((i64)ex[w] << 16) / tot; pb[w] = (i32)p; if (p > pm) pm = p; }
    if (pm <= 32767) { for (int w = 0; w < W; w++) phi[w] = (i16)pb[w]; return 1; }
    for (int w = 0; w < W; w++) { phi[w] = (i16)(pb[w] >> 8); plo[w] = (i16)(pb[w] & 255); }
    return 0;
#endif
}

/* o[i] = (sum_w p_w V[w][i]) >> 16 for one head (dense: real attention is dense). one = 1: phi holds p (<= 32767), one
 * int32 pass (|sum| <= 2^16 x 2^15); else phi = p >> 8 (<= 256), plo = p & 255: two int32 passes (|sum| <= 2^23 and
 * < 2^30), o = ((H << 8) + L) >> 16. */
static void head_pv(const tfmodel *m, const i16 *Vb, const i16 *phi, const i16 *plo, int one, i32 *o)
{
    const int W = m->W, dh = m->dh;
#ifdef TFS_AVX512
    if (one) {
        for (int dv = 0; dv < dh; dv += 16) {
            __m512i ah[8];
            for (int u = 0; u < 8; u++) ah[u] = _mm512_setzero_si512();
            int w2 = 0;
            for (; w2 + 8 <= W / 2; w2 += 8)
                for (int u = 0; u < 8; u++) {
                    i32 ph; memcpy(&ph, phi + 2 * (w2 + u), 4);
                    ah[u] = _mm512_dpwssd_epi32(ah[u], _mm512_set1_epi32(ph), _mm512_loadu_si512((const void *)(Vb + ((size_t)(w2 + u) * dh + dv) * 2)));
                }
            for (; w2 < W / 2; w2++) {
                i32 ph; memcpy(&ph, phi + 2 * w2, 4);
                ah[0] = _mm512_dpwssd_epi32(ah[0], _mm512_set1_epi32(ph), _mm512_loadu_si512((const void *)(Vb + ((size_t)w2 * dh + dv) * 2)));
            }
            const __m512i H = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(ah[0], ah[1]), _mm512_add_epi32(ah[2], ah[3])),
                                               _mm512_add_epi32(_mm512_add_epi32(ah[4], ah[5]), _mm512_add_epi32(ah[6], ah[7])));
            _mm512_storeu_si512((void *)(o + dv), _mm512_srai_epi32(H, 16));
        }
        return;
    }
    for (int dv = 0; dv < dh; dv += 16) {
        __m512i ah[4], al[4];
        for (int u = 0; u < 4; u++) { ah[u] = _mm512_setzero_si512(); al[u] = _mm512_setzero_si512(); }
        int w2 = 0;
        for (; w2 + 4 <= W / 2; w2 += 4)
            for (int u = 0; u < 4; u++) {
                i32 ph, pl; memcpy(&ph, phi + 2 * (w2 + u), 4); memcpy(&pl, plo + 2 * (w2 + u), 4);
                const __m512i v = _mm512_loadu_si512((const void *)(Vb + ((size_t)(w2 + u) * dh + dv) * 2));
                ah[u] = _mm512_dpwssd_epi32(ah[u], _mm512_set1_epi32(ph), v);
                al[u] = _mm512_dpwssd_epi32(al[u], _mm512_set1_epi32(pl), v);
            }
        for (; w2 < W / 2; w2++) {
            i32 ph, pl; memcpy(&ph, phi + 2 * w2, 4); memcpy(&pl, plo + 2 * w2, 4);
            const __m512i v = _mm512_loadu_si512((const void *)(Vb + ((size_t)w2 * dh + dv) * 2));
            ah[0] = _mm512_dpwssd_epi32(ah[0], _mm512_set1_epi32(ph), v);
            al[0] = _mm512_dpwssd_epi32(al[0], _mm512_set1_epi32(pl), v);
        }
        const __m512i H = _mm512_add_epi32(_mm512_add_epi32(ah[0], ah[1]), _mm512_add_epi32(ah[2], ah[3]));
        const __m512i L = _mm512_add_epi32(_mm512_add_epi32(al[0], al[1]), _mm512_add_epi32(al[2], al[3]));
        /* ((H << 8) + L) >> 16 in int64 */
        const __m512i Hl = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(H)), Hh = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(H, 1));
        const __m512i Ll = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(L)), Lh = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(L, 1));
        const __m512i ol = _mm512_srai_epi64(_mm512_add_epi64(_mm512_slli_epi64(Hl, 8), Ll), 16);
        const __m512i oh = _mm512_srai_epi64(_mm512_add_epi64(_mm512_slli_epi64(Hh, 8), Lh), 16);
        _mm256_storeu_si256((__m256i *)(o + dv), _mm512_cvtepi64_epi32(ol));
        _mm256_storeu_si256((__m256i *)(o + dv + 8), _mm512_cvtepi64_epi32(oh));
    }
#else
    if (one) {
        for (int i = 0; i < dh; i++) {
            i32 H = 0;
            for (int w2 = 0; w2 < W / 2; w2++) {
                const i16 *v = Vb + ((size_t)w2 * dh + i) * 2;
                H += (i32)phi[2 * w2] * v[0] + (i32)phi[2 * w2 + 1] * v[1];
            }
            o[i] = H >> 16;
        }
        return;
    }
    for (int i = 0; i < dh; i++) {
        i32 H = 0, L = 0;
        for (int w2 = 0; w2 < W / 2; w2++) {
            const i16 *v = Vb + ((size_t)w2 * dh + i) * 2;
            H += (i32)phi[2 * w2] * v[0] + (i32)phi[2 * w2 + 1] * v[1];
            L += (i32)plo[2 * w2] * v[0] + (i32)plo[2 * w2 + 1] * v[1];
        }
        o[i] = (i32)(((i64)H * 256 + L) >> 16);
    }
#endif
}

/* ------------------------------------------------------------------------------------------------ the fast block step */
/* R <= RB fast curves cid[0..R) at model time T; x rows (int64, NF); out rows (3K). ok[r] = 0 when curve r must be
 * promoted and re-run on the reference path (its outputs here are discarded). */
static int fast_block(tfstate *s, int R, const i64 *cid, i64 T, const i64 *const *xrow, i64 *const *orow, int *ok)
{
    const tfmodel *m = s->m;
    const int D = m->D, W = m->W, dh = m->dh, NH = m->NH, K = m->K, FF = m->FF;
    const int slot = (int)(T % W);
    const int ldm = s->ldm;
    i32 *h = s->h, *a = s->a, *o = s->o, *u = s->u, *f = s->f;
    i64 *acc = s->acc;
    /* embed: x (int64) -> int32 rows in a */
    for (int r = 0; r < R; r++) {
        ok[r] = 1;
        for (int i = 0; i < m->NF; i++) {
            const i64 v = xrow[r][i];
            if (v < -((i64)1 << 30) || v > ((i64)1 << 30)) return -2;
            a[(size_t)r * ldm + i] = (i32)v;
        }
    }
    matvec(s, &m->m0, R, a, ldm, acc, ldm);
    for (int r = 0; r < R; r++)
        post_clamp(acc + (size_t)r * ldm, m->m0.b, D, m->ws0 - (int)m->aq, 0, (i64)1 << 24, (int)(m->hq - m->aq), h + (size_t)r * D);
    for (int l = 0; l < m->NL; l++) {
        const i32 *relL = s->relslot + (size_t)l * NH * W;
        for (int r = 0; r < R; r++) ln_row(m, h + (size_t)r * D, a + (size_t)r * ldm);
        const tfmat *mq = &m->mqkv[l];
        matvec(s, mq, R, a, ldm, acc, ldm);
        const int shq = m->wsq[l] + (int)(m->hq - m->aq);
        for (int r = 0; r < R; r++) {
            if (!ok[r]) continue;
            const i64 c = cid[r];
            const i64 *ar = acc + (size_t)r * ldm;
            i32 *qr = s->qkv + (size_t)r * 3 * D;
            i64 qmn, qmx, kmn, kmx;
            post_range(ar, mq->b, D, shq, qr, &qmn, &qmx);
            post_range(ar + D, mq->b + D, 2 * D, shq, qr + D, &kmn, &kmx);
            if (qmn < INT32_MIN || qmx > INT32_MAX || kmn < -32768 || kmx > 32767) { ok[r] = 0; continue; }
            /* write k, v at the slot; running max ||k_h||^2 */
            u64 km = s->kmax2[(size_t)c * m->NL + l];
            for (int hh = 0; hh < NH; hh++) {
                i16 *Kb = kblk(s, c, l, hh), *Vb = Kb + (size_t)dh * W;
                const i32 *kk = qr + D + hh * dh, *vv = qr + 2 * D + hh * dh;
                u64 kn = 0;
                for (int i = 0; i < dh; i++) {
                    Kb[((size_t)(i >> 1) * W + slot) * 2 + (i & 1)] = (i16)kk[i];
                    Vb[((size_t)(slot >> 1) * dh + i) * 2 + (slot & 1)] = (i16)vv[i];
                    kn += (u64)((i64)kk[i] * kk[i]);
                }
                if (kn > km) km = kn;
            }
            s->kmax2[(size_t)c * m->NL + l] = km;
            /* attention: every phase for all heads of the curve (independent chains overlap) */
            i32 *dots = s->dots, *lg = s->lg, *ex = s->ex;
            i16 *phi = s->phi, *plo = s->plo;
            i32 mxh[MAXH], toth[MAXH]; int oneh[MAXH];
            for (int hh = 0; hh < NH; hh++) {
                const i16 *Kb = kblk(s, c, l, hh);
                const i32 *q = qr + hh * dh;
                i16 *q16 = s->q16 + (size_t)hh * dh;
                u64 qn = 0; int qbig = 0;
                for (int i = 0; i < dh; i++) {
                    if (q[i] < -32768 || q[i] > 32767) qbig = 1;
                    else qn += (u64)((i64)q[i] * q[i]);
                    q16[i] = (i16)q[i];
                }
                const int exact32 = !qbig && ((unsigned __int128)qn * km <= (unsigned __int128)LIM62);
                i32 *dh_ = dots + (size_t)hh * W;
                if (exact32) head_scores(m, Kb, q16, dh_);
                else {
                    /* scalar int64 dots; store lv << 8 so that sm_logits' >> 8 gives lv = d >> 8 exactly */
                    for (int w = 0; w < W; w++) {
                        i64 d = 0;
                        for (int i = 0; i < dh; i++) d += (i64)q[i] * Kb[((size_t)(i >> 1) * W + w) * 2 + (i & 1)];
                        const i64 lv = d >> 8;
                        if (lv < -(1 << 22) || lv > (1 << 22)) return -2;
                        dh_[w] = (i32)(lv * 256);
                    }
                }
            }
            for (int hh = 0; hh < NH; hh++) mxh[hh] = sm_logits(m, dots + (size_t)hh * W, relL + (size_t)hh * W, lg + (size_t)hh * W);
            for (int hh = 0; hh < NH; hh++) toth[hh] = sm_exp(m, lg + (size_t)hh * W, mxh[hh], ex + (size_t)hh * W);
            for (int hh = 0; hh < NH; hh++)
                oneh[hh] = sm_p(m, ex + (size_t)hh * W, toth[hh], lg + (size_t)hh * W, phi + (size_t)hh * W, plo + (size_t)hh * W);
            for (int hh = 0; hh < NH; hh++) {
                const i16 *Vb = kblk(s, c, l, hh) + (size_t)dh * W;
                head_pv(m, Vb, phi + (size_t)hh * W, plo + (size_t)hh * W, oneh[hh], o + (size_t)r * D + hh * dh);
            }
        }
        /* h += (Wo o + bo) >> (wso - 4) */
        const tfmat *mo = &m->mo[l];
        matvec(s, mo, R, o, D, acc, ldm);
        const int sho = m->wso[l] - (int)(m->hq - m->aq);
        for (int r = 0; r < R; r++) post_resid(acc + (size_t)r * ldm, mo->b, D, sho, m->h_lim, h + (size_t)r * D);
        for (int r = 0; r < R; r++) ln_row(m, h + (size_t)r * D, a + (size_t)r * ldm);
        const tfmat *m1 = &m->m1[l], *m2 = &m->m2[l];
        matvec(s, m1, R, a, ldm, acc, ldm);
        const int sh1 = m->ws1[l] + (int)(m->hq - m->aq);
        for (int r = 0; r < R; r++) post_clamp(acc + (size_t)r * ldm, m1->b, FF, sh1, 0, (i64)1 << 24, 0, u + (size_t)r * ldm);
        matvec(s, m2, R, u, ldm, acc, ldm);
        const int sh2 = m->ws2[l] - (int)(m->hq - m->aq);
        for (int r = 0; r < R; r++) post_resid(acc + (size_t)r * ldm, m2->b, D, sh2, m->h_lim, h + (size_t)r * D);
    }
    matvec(s, &m->mf, R, h, D, acc, ldm);
    const int shf = m->wsf + (int)(m->hq - m->aq);
    for (int r = 0; r < R; r++) post_clamp(acc + (size_t)r * ldm, m->mf.b, m->FD, shf, 0, (i64)1 << 24, 0, f + (size_t)r * ldm);
    matvec(s, &m->mh, R, f, ldm, acc, ldm);
    for (int r = 0; r < R; r++) {
        if (!ok[r]) continue;
        i64 *out = orow[r];
        for (int j = 0; j < 3 * K; j++) {
            const i64 v = (acc[(size_t)r * ldm + j] + m->mh.b[j]) >> m->wsh;
            out[j] = j >= 2 * K ? clampi(v, m->logs_lo, m->logs_hi) : v;
        }
    }
    return 0;
}

static void fill_relslot(tfstate *s, i64 T)
{
    const tfmodel *m = s->m;
    const int W = m->W, slot = (int)(T % W);
    const int full = T >= W - 1;
    for (int l = 0; l < m->NL; l++)
        for (int hh = 0; hh < m->NH; hh++) {
            const i32 *rel = m->rel + ((size_t)l * m->NH + hh) * W;
            i32 *rs = s->relslot + ((size_t)l * m->NH + hh) * W;
            /* offset of ring slot w = (T - w) mod W = slot - w (+ W when negative); valid when offset <= T */
            for (int w = 0; w <= slot; w++) rs[w] = (full || slot - w <= T) ? rel[slot - w] : -(1 << 29);
            for (int w = slot + 1; w < W; w++) rs[w] = (full || slot - w + W <= T) ? rel[slot - w + W] : -(1 << 29);
        }
}

/* the n curves ids[] at model time T; curve i's features / outputs are row rows[i] (rows == NULL: row i) of x / out;
 * relslot must be filled for T. No contract checks. */
static int run_rows(tfstate *s, i64 T, i64 n, const i64 *ids, const i64 *rows, const i64 *x, i64 *out)
{
    const tfmodel *m = s->m;
    const int NF = m->NF, K3 = 3 * m->K;
    i64 cid[RB]; const i64 *xr[RB]; i64 *orow[RB]; int ok[RB]; i64 idx[RB];
    int R = 0;
    /* curves in alternating order (ascending on even steps, descending on odd ones): the curves whose caches were used
     * last come first (more L2 hits when the active caches exceed L2); the rows are independent. */
    const int rev = (int)(T & 1);
    for (i64 ii = 0; ii <= n; ii++) {
        const i64 i = rev && ii < n ? n - 1 - ii : ii;
        if (ii < n) {
            const i64 c = ids[i], row = rows ? rows[i] : i;
            if (!m->fast || s->wide[c]) {
                if (!s->wide[c] && promote(s, c)) return -2;
                const int rc = ref_step(s, c, T, x + row * NF, out + row * K3);
                if (rc) return rc;
                continue;
            }
            cid[R] = c; xr[R] = x + row * NF; orow[R] = out + row * K3; idx[R] = row; R++;
            if (R < RB) continue;
        }
        if (R == 0) continue;
        const int rc = fast_block(s, R, cid, T, xr, orow, ok);
        if (rc) return rc;
        for (int r = 0; r < R; r++)
            if (!ok[r]) {
                if (promote(s, cid[r])) return -2;
                const int rc2 = ref_step(s, cid[r], T, x + idx[r] * NF, out + idx[r] * K3);
                if (rc2) return rc2;
            }
        R = 0;
    }
    return 0;
}

/* ------------------------------------------------------------------------------------------------ worker threads */
static inline void cpu_relax(void)
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

typedef struct { tfstate *s; int k; } wkarg;

static void *pool_worker(void *arg)
{
    wkarg *wa = (wkarg *)arg;
    tfpool *P = wa->s->pool;
    const int k = wa->k;
    free(wa);
    if (P->cpus[k] >= 0) {
#ifdef __linux__
        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(P->cpus[k], &cs);
        pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#endif
    }
    long seen = 0;
    for (;;) {
        long spins = 0, g;
        while ((g = atomic_load_explicit(&P->gen, memory_order_acquire)) == seen) {
            if (atomic_load_explicit(&P->quit, memory_order_relaxed)) return NULL;
            if (++spins < 200000) cpu_relax();
            else if (spins < 200100) sched_yield();
            else { struct timespec ts = {0, 20000}; nanosleep(&ts, NULL); }       /* idle: stop burning the core */
        }
        seen = g;
        if (atomic_load_explicit(&P->quit, memory_order_relaxed)) return NULL;
        P->rc[k] = run_rows(P->sub[k], P->T, P->n[k], P->ids[k], P->rows[k], P->x, P->out);
        atomic_fetch_add_explicit(&P->done, 1, memory_order_release);
    }
}

static void pool_stop(tfstate *s)
{
    tfpool *P = s->pool;
    if (!P || s->is_clone) return;
    atomic_store_explicit(&P->quit, 1, memory_order_release);
    atomic_fetch_add_explicit(&P->gen, 1, memory_order_release);
    for (int k = 1; k <= P->nw; k++) pthread_join(P->th[k], NULL);
    for (int k = 1; k <= P->nw; k++) tfs_state_free(P->sub[k]);
    for (int k = 0; k <= P->nw; k++) { free(P->ids[k]); free(P->rows[k]); }
    free(P);
    s->pool = NULL;
}

/* nthreads >= 2: nthreads - 1 worker threads (+ the caller) share every step's active curves, curve c on thread
 * c % nthreads (its cache stays in that core's L2); cpus (optional, nthreads - 1 entries): pin worker k to cpus[k - 1].
 * Workers spin between steps (then sleep after ~200k spins). nthreads <= 1 stops the pool. Returns 0 or -1. */
int tfs_set_threads(void *vs, int nthreads, const int *cpus)
{
    tfstate *s = (tfstate *)vs;
    if (!s || s->is_clone) return -1;
    pool_stop(s);
    if (nthreads <= 1) return 0;
    if (nthreads > TMAX) nthreads = TMAX;
    tfpool *P = (tfpool *)calloc(1, sizeof(tfpool));
    if (!P) return -1;
    P->nw = nthreads - 1;
    P->sub[0] = s;
    const i64 Cc = s->C > 0 ? s->C : 1;
    for (int k = 0; k <= P->nw; k++) {
        P->ids[k] = (i64 *)malloc((size_t)Cc * sizeof(i64)); P->rows[k] = (i64 *)malloc((size_t)Cc * sizeof(i64));
        P->cpus[k] = (k >= 1 && cpus) ? cpus[k - 1] : -1;
        if (!P->ids[k] || !P->rows[k]) return -1;
    }
    for (int k = 1; k <= P->nw; k++) {
        tfstate *c = (tfstate *)malloc(sizeof(tfstate));
        if (!c) return -1;
        *c = *s; c->is_clone = 1; c->pool = NULL;
        if (alloc_scratch(c)) return -1;
        P->sub[k] = c;
    }
    s->pool = P;
    for (int k = 1; k <= P->nw; k++) {
        wkarg *wa = (wkarg *)malloc(sizeof(wkarg));
        wa->s = s; wa->k = k;
        if (pthread_create(&P->th[k], NULL, pool_worker, wa)) { P->nw = k - 1; pool_stop(s); return -1; }
    }
    return 0;
}

/* one step at model time T for the n curves ids[] (rows in order; no contract checks) */
static int step_core(tfstate *s, i64 T, i64 n, const i64 *ids, const i64 *x, i64 *out)
{
    const tfmodel *m = s->m;
    if (m->fast) fill_relslot(s, T);
    tfpool *P = s->pool;
    if (!P || n < 2 * (P->nw + 1)) return run_rows(s, T, n, ids, NULL, x, out);
    const int nt = P->nw + 1;
    for (int k = 0; k < nt; k++) P->n[k] = 0;
    for (i64 i = 0; i < n; i++) {
        const int k = (int)(ids[i] % nt);
        P->ids[k][P->n[k]] = ids[i]; P->rows[k][P->n[k]] = i; P->n[k]++;
    }
    P->T = T; P->x = x; P->out = out;
    atomic_store_explicit(&P->done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&P->gen, 1, memory_order_release);
    const int rc0 = run_rows(s, T, P->n[0], P->ids[0], P->rows[0], x, out);
    long spins = 0;
    while (atomic_load_explicit(&P->done, memory_order_acquire) < P->nw) {
        if (++spins < 100000) cpu_relax();
        else sched_yield();                          /* oversubscribed cores: let the workers run */
    }
    if (rc0) return rc0;
    for (int k = 1; k < nt; k++) if (P->rc[k]) return P->rc[k];
    return 0;
}

/* the pad template: one curve after `pad` zero-feature steps from empty caches (IntTf.start(1, pad)) */
static int run_pad_template(tfmodel *m, i64 pad)
{
    tfstate *s = state_alloc(m, 1);
    if (!s) return -2;
    s->pad = 0;
    i64 *x = (i64 *)calloc((size_t)m->NF, sizeof(i64)), out[3 * 64];
    const i64 id = 0;
    int rc = 0;
    if (!x || 3 * m->K > 3 * 64) rc = -2;
    for (i64 T = 0; T < pad && !rc; T++) rc = step_core(s, T, 1, &id, x, out);
    free(x);
    if (rc) { tfs_state_free(s); return rc; }
    free(m->tpl16); free(m->tplw); m->tpl16 = NULL; m->tplw = NULL;
    m->tpl16 = (i16 *)malloc((size_t)m->curve_bytes);
    if (!m->tpl16) { tfs_state_free(s); return -2; }
    memcpy(m->tpl16, s->c16, (size_t)m->curve_bytes);
    m->tpl_wide = s->wide[0] != NULL;
    if (m->tpl_wide) {
        m->tplw = (i32 *)malloc((size_t)wide_len(m) * sizeof(i32));
        if (!m->tplw) { tfs_state_free(s); return -2; }
        memcpy(m->tplw, s->wide[0], (size_t)wide_len(m) * sizeof(i32));
    }
    for (int l = 0; l < m->NL; l++) m->tpl_kmax2[l] = s->kmax2[l];
    m->tpl_pad = pad; m->tpl_ok = 1;
    tfs_state_free(s);
    return 0;
}

void *tfs_state_create(void *vm, i64 C, i64 pad)
{
    tfmodel *m = (tfmodel *)vm;
    if (!m || C < 0 || pad < 0) return NULL;
    if (pad > 0 && !(m->tpl_ok && m->tpl_pad == pad))
        if (run_pad_template(m, pad)) return NULL;
    tfstate *s = state_alloc(m, C > 0 ? C : 1);
    if (!s) return NULL;
    s->C = C; s->pad = pad; s->t = 0;
    if (pad > 0) {
        for (i64 c = 0; c < C; c++) {
            memcpy(s->c16 + (size_t)c * (m->curve_bytes / sizeof(i16)), m->tpl16, (size_t)m->curve_bytes);
            for (int l = 0; l < m->NL; l++) s->kmax2[(size_t)c * m->NL + l] = m->tpl_kmax2[l];
            if (m->tpl_wide) {
                s->wide[c] = (i32 *)malloc((size_t)wide_len(m) * sizeof(i32));
                if (!s->wide[c]) { tfs_state_free(s); return NULL; }
                memcpy(s->wide[c], m->tplw, (size_t)wide_len(m) * sizeof(i32));
            }
        }
    }
    return s;
}

int tfs_step(void *vs, i64 t, i64 n_act, const i64 *act, const i64 *x, i64 *out)
{
    tfstate *s = (tfstate *)vs;
    if (!s || t != s->t || n_act < 0 || n_act > s->C) return -1;
    for (i64 i = 0; i < n_act; i++) {
        const i64 c = act[i];
        if (c < 0 || c >= s->C || (i && c <= act[i - 1]) || s->last[c] != t - 1) return -1;
    }
    const int rc = step_core(s, s->pad + t, n_act, act, x, out);
    if (rc) return rc;
    for (i64 i = 0; i < n_act; i++) s->last[act[i]] = t;
    s->t = t + 1;
    return 0;
}

/* The encoder's model pass (codec.CurveCodec._pack_residuals): every residual of a clip at once. The C columns run
 * the lockstep of the decoder -- step t codes the columns with n[c] > t, ascending -- so x / out hold the rows in that
 * time-major order ([N, NF] / [N, 3K], N = sum n). One state for the clip, no Python between the steps; the rows equal
 * IntTf.step's (the same step as tfs_step). Returns 0 or the step's error code. */
int tfs_set_threads(void *vs, int nthreads, const int *cpus);

int tfs_forward_lockstep(void *vm, i64 C, i64 pad, const i64 *n, const i64 *x, i64 *out, int nthreads, const int *cpus)
{
    tfmodel *m = (tfmodel *)vm;
    tfstate *s = (tfstate *)tfs_state_create(vm, C, pad);
    if (!s) return -2;
    if (nthreads > 1 && tfs_set_threads(s, nthreads, cpus)) { tfs_state_free(s); return -2; }
    i64 *ids = (i64 *)malloc((size_t)(C > 0 ? C : 1) * sizeof(i64));
    if (!ids) { tfs_state_free(s); return -2; }
    i64 nmax = 0;
    for (i64 c = 0; c < C; c++) if (n[c] > nmax) nmax = n[c];
    i64 off = 0; int rc = 0;
    const int NF = m->NF, K3 = 3 * m->K;
    for (i64 t = 0; t < nmax && !rc; t++) {
        i64 k = 0;
        for (i64 c = 0; c < C; c++) if (n[c] > t) ids[k++] = c;
        rc = step_core(s, pad + t, k, ids, x + (size_t)off * NF, out + (size_t)off * K3);
        off += k;
    }
    free(ids);
    tfs_state_free(s);
    return rc;
}

int tfs_simd(void)
{
#ifdef TFS_AVX512
    return 512;
#else
    return 0;
#endif
}
