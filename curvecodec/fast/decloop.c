/* The decoder of the blob in C, bit-exact with CurveCodec.decode (curvecodec/fast/decoder.py builds, loads and drives
 * it):
 *
 *   dl_dec        the blob's rANS stream with the adaptive context model (rans.RansDecoder) and the raw MSB-first bit
 *                 stream (rans.BitReader)
 *   dl_v2_*       dl_v2_header (every header loop after the type symbols, in the Python decoder's order), dl_v2_columns
 *                 (the coded columns in coder order + the static features of features.CurveState), dl_v2_lockstep,
 *                 dl_v2_interp + dl_v2_write_rot_np / _trans (dequantize, interpolate, exp_map, the float32 clip)
 *   dl_mix        the sign-magnitude alphabet + the mixture -> rANS frequency tables (features.symbol_freqs) at the
 *                 alphabet's distinct CDF points only, the sigmoid's saturated ranges from precomputed thresholds
 *   dl_lockstep   the time-major residual loop over the coded columns: the 36 integer features (features.CurveState),
 *                 the model through a dl_step_fn hook (decloop.h; the transformer step of tfstep.c), the frequency
 *                 tables, the rANS symbol + raw low bits + alphabet value, the incremental predictors (predict.IncPred:
 *                 d1 / d2 / d2t / NLMS), the decoded integers of every column
 *
 * Integer semantics mirror numpy int64 (wrap-around +, -, *, <<; arithmetic >>; floor //). The float operations are the
 * ones numpy does, in its order (the interpolation, d2t's ratio / product / rint, exp_map around numpy's own sin / cos
 * loops), compiled without contraction, or exact integer arithmetic in doubles -- see the notes at each use.
 */
#define _POSIX_C_SOURCE 200112L
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "decloop.h"
#if defined(__AVX512F__) && defined(__AVX512DQ__)
#include <immintrin.h>
#define DL_AVX512 1
#else
#define DL_AVX512 0
#endif

typedef int64_t i64;
typedef uint64_t u64;

#define NF DL_NFEAT
#define HH 8
#define NTAU 5

enum { DL_OK = 0, DL_E_ALLOC = -2, DL_E_EOF_RANS = -10, DL_E_EOF_RAW = -11, DL_E_CTX = -12, DL_E_RANGE = -13, DL_E_ARG = -14,
       DL_E_MODEL = -1000 };

static inline i64 wadd(i64 a, i64 b) { return (i64)((u64)a + (u64)b); }
static inline i64 wsub(i64 a, i64 b) { return (i64)((u64)a - (u64)b); }
static inline i64 wmul(i64 a, i64 b) { return (i64)((u64)a * (u64)b); }
static inline i64 shl(i64 a, int s) { return (i64)((u64)a << s); }
static inline i64 asr(i64 a, int s) { return a >> s; }
static inline i64 floordiv(i64 a, i64 b)
{
    if (b == -1) return (i64)(0 - (u64)a);
    i64 q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1;
    return q;
}
static inline i64 clipi(i64 v, i64 lo, i64 hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline int bitlen(u64 v) { return v ? 64 - __builtin_clzll(v) : 0; }
static inline i64 iabs(i64 v) { return v < 0 ? (i64)(0 - (u64)v) : v; }


/* ================================================================================================================ */
/* the header stream: adaptive rANS (rans.RansDecoder) + raw bits (rans.BitReader)                            */
/* ================================================================================================================ */
typedef struct {
    uint8_t *rd; i64 rn, rp; u64 x;                  /* rANS bytes, position, state */
    i64 nctx, nsym, M, SB, L, scale, limit, inc;
    i64 *cnt, *tot, *cum;
    uint8_t *bd; i64 bn, bp; u64 acc; int nacc;      /* raw bits */
    int err;
} dl_dec;

void dl_close(dl_dec *d)
{
    if (!d) return;
    free(d->rd); free(d->bd); free(d->cnt); free(d->tot); free(d->cum); free(d);
}

/* init [nctx, nsym] int64 initial counts; scale_bits = RANS_SCALE_BITS, L = RANS_L, inc = MODEL_INC, adapt_rate as RansDec */
dl_dec *dl_open(const uint8_t *rans, i64 nr, const i64 *init, i64 nctx, i64 nsym, i64 scale_bits, i64 L, i64 inc, i64 adapt_rate,
                const uint8_t *raw, i64 nraw)
{
    dl_dec *d = (dl_dec *)calloc(1, sizeof(dl_dec));
    if (!d) return 0;
    d->rd = (uint8_t *)malloc(nr > 0 ? (size_t)nr : 1); d->bd = (uint8_t *)malloc(nraw > 0 ? (size_t)nraw : 1);
    d->cnt = (i64 *)malloc(sizeof(i64) * (size_t)(nctx * nsym)); d->tot = (i64 *)malloc(sizeof(i64) * (size_t)nctx);
    d->cum = (i64 *)malloc(sizeof(i64) * (size_t)nsym);
    if (!d->rd || !d->bd || !d->cnt || !d->tot || !d->cum) { dl_close(d); return 0; }
    if (nr > 0) memcpy(d->rd, rans, (size_t)nr);
    if (nraw > 0) memcpy(d->bd, raw, (size_t)nraw);
    d->rn = nr; d->bn = nraw;
    d->x = 0; d->rp = 4;
    if (nr >= 4) d->x = (u64)d->rd[0] | ((u64)d->rd[1] << 8) | ((u64)d->rd[2] << 16) | ((u64)d->rd[3] << 24);
    d->nctx = nctx; d->nsym = nsym; d->SB = scale_bits; d->M = (i64)1 << scale_bits; d->L = L; d->inc = inc;
    d->scale = d->M - nsym; d->limit = nsym + (inc << (adapt_rate + 5));
    for (i64 c = 0; c < nctx; c++) {
        i64 t = 0;
        for (i64 j = 0; j < nsym; j++) {
            i64 v = init[c * nsym + j];
            if (v < 0 || v > ((i64)1 << 30)) { dl_close(d); return 0; }       /* keeps cnt * scale < 2^53 (hdr_decode) */
            d->cnt[c * nsym + j] = v; t += v;
        }
        if (t <= 0) { dl_close(d); return 0; }
        d->tot[c] = t;
    }
    return d;
}

/* RansDec.decode(c) */
static i64 hdr_decode(dl_dec *d, i64 c)
{
    if (d->err) return 0;
    if (c < 0 || c >= d->nctx) { d->err = DL_E_CTX; return 0; }
    const i64 nsym = d->nsym, scale = d->scale;
    i64 *cnt = d->cnt + c * nsym, *q = d->cum;
    i64 tot = d->tot[c];
    /* q_j = floor(cnt_j scale / tot) (cnt_j scale < 2^53): trunc(a (1 / tot)) is within one of it, one correction each
     * way makes it exact; all q first (vectorised), then the cumulative search stops at the symbol */
    const double inv = 1.0 / (double)tot;
    i64 qsum = 0;
    for (i64 j = 0; j < nsym; j++) {
        const i64 a = cnt[j] * scale;
        i64 v = (i64)((double)a * inv);
        v -= v * tot > a;
        v += (v + 1) * tot <= a;
        q[j] = v; qsum += v;
    }
    const i64 rem = d->M - (qsum + nsym);
    const i64 slot = (i64)(d->x & (u64)(d->M - 1));
    const i64 v = slot - rem;
    i64 s = 0, Cs = q[0] + 1, Cp = 0;                                     /* Cum_s = sum_{i <= s} q_i + (s + 1) */
    while (Cs <= v) {                                                     /* searchsorted(Cum, slot - rem, 'right') */
        if (++s >= nsym) { d->err = DL_E_RANGE; return 0; }
        Cp = Cs; Cs += q[s] + 1;
    }
    i64 st, f;
    if (s == 0) { st = 0; f = Cs + rem; }                                 /* q0 + 1 + rem */
    else { st = Cp + rem; f = Cs - Cp; }                                  /* q_s + 1 */
    u64 x = (u64)f * (d->x >> d->SB) + (u64)(slot - st);
    while (x < (u64)d->L) {
        if (d->rp >= d->rn) { d->err = DL_E_EOF_RANS; return 0; }
        x = (x << 8) | d->rd[d->rp++];
    }
    d->x = x;
    cnt[s] += d->inc; tot += d->inc;
    if (tot > d->limit) {
        i64 t2 = 0;
        for (i64 j = 0; j < nsym; j++) { cnt[j] = (cnt[j] + 1) >> 1; t2 += cnt[j]; }
        tot = t2;
    }
    d->tot[c] = tot;
    return s;
}

/* BitReader.read(n), n in 0 .. 64 */
static u64 br_read(dl_dec *d, int n)
{
    if (n <= 0 || d->err) return 0;
    if (n > 56) { u64 hi = br_read(d, n - 32); u64 lo = br_read(d, 32); return (hi << 32) | lo; }
    while (d->nacc < n) {
        if (d->bp >= d->bn) { d->err = DL_E_EOF_RAW; return 0; }
        d->acc = (d->acc << 8) | d->bd[d->bp++]; d->nacc += 8;
    }
    d->nacc -= n;
    u64 v = (d->acc >> d->nacc) & ((n == 64) ? ~(u64)0 : (((u64)1 << n) - 1));
    d->acc &= d->nacc ? (((u64)1 << d->nacc) - 1) : 0;
    return v;
}

/* BitReader.read_exp_golomb(k) */
static u64 br_expg(dl_dec *d, int k)
{
    i64 z = 0;
    for (;;) {
        if (d->err) return 0;
        if (d->nacc == 0) {
            if (d->bp >= d->bn) { d->err = DL_E_EOF_RAW; return 0; }
            d->acc = d->bd[d->bp++]; d->nacc = 8;
        }
        if (d->acc == 0) { z += d->nacc; d->acc = 0; d->nacc = 0; if (z > 62) { d->err = DL_E_RANGE; return 0; } continue; }
        int leadz = d->nacc - bitlen(d->acc);
        z += leadz;
        d->nacc -= leadz + 1;
        d->acc &= d->nacc ? (((u64)1 << d->nacc) - 1) : 0;
        break;
    }
    if (z + k > 62) { d->err = DL_E_RANGE; return 0; }
    const int zk = (int)(z + k);
    return (((u64)1 << zk) | br_read(d, zk)) - ((u64)1 << k);
}

static inline i64 unzigzag(u64 u) { return (i64)((u >> 1) ^ (0 - (u & 1))); }

/* n symbols of one context (types, modes, predictor ids) */
int dl_syms(dl_dec *d, i64 ctx, i64 n, i64 *out)
{
    for (i64 i = 0; i < n; i++) out[i] = hdr_decode(d, ctx);
    return d->err;
}

/* _GridContainer._get_mags per value with its context: c = symbol, m = read(max(c - 1, 0)), mag_join, unzigzag */
static i64 get_mag(dl_dec *d, i64 ctx)
{
    i64 c = hdr_decode(d, ctx);
    if (c > 63) { d->err = DL_E_RANGE; return 0; }
    u64 m = br_read(d, (int)(c > 1 ? c - 1 : 0));
    u64 u = c > 0 ? (((u64)1 << (c - 1)) | m) : 0;
    return unzigzag(u);
}

/* One keyframe / per-component track's key set: the gap loop (EM gap contexts kbase + pc, escapes through exp-Golomb)
 * and, for a per-component track, the presence flags (contexts fb ..).  U [cap] key frames, pres [cap, 3] (0 / 1).
 * Returns the number of keys, or a negative error. */
i64 dl_keys(dl_dec *d, i64 F, i64 kbase, i64 gcap, i64 esc, i64 is_ck, i64 fb, i64 cap, i64 *U, uint8_t *pres)
{
    i64 n = 1; U[0] = 0; i64 pc = 0;
    while (U[n - 1] < F - 1) {
        if (n >= cap) { d->err = DL_E_RANGE; return d->err; }
        i64 s = hdr_decode(d, kbase + pc);
        i64 gm = s < esc ? s : wadd(esc, (i64)br_expg(d, 0));
        if (d->err) return d->err;
        U[n] = wadd(U[n - 1], wadd(gm, 1)); n++;
        i64 bl = bitlen((u64)gm);
        pc = 1 + (bl < gcap ? bl : gcap);
    }
    if (is_ck) {
        for (i64 j = 0; j < 3 * n; j++) pres[j] = 1;
        i64 p0 = 1, p1 = 1, p2 = 1;
        for (i64 j = 1; j < n - 1; j++) {
            i64 f0 = hdr_decode(d, fb + p0);
            i64 f1 = hdr_decode(d, fb + 4 + p1 * 2 + f0);
            i64 f2 = (f0 == 0 && f1 == 0) ? 1 : hdr_decode(d, fb + 12 + p2 * 4 + f0 * 2 + f1);
            if (d->err) return d->err;
            pres[3 * j] = f0 != 0; pres[3 * j + 1] = f1 != 0; pres[3 * j + 2] = f2 != 0;
            p0 = f0; p1 = f1; p2 = f2;
        }
    }
    return d->err ? d->err : n;
}

/* ================================================================================================================ */
/* the mixture -> frequency tables (features.symbol_freqs = mixture_weights_q16 + cdf_q16 + normalisation)            */
/* ================================================================================================================ */
typedef struct {
    i64 S, U;                    /* alphabet symbols, distinct CDF points */
    i64 *lo, *hi, *wd;           /* alphabet (sign-magnitude, features.make_alphabet) */
    i64 *dq;                     /* [U] clip((pt << 8) + 128, +-MU_LIM) of the sorted distinct points pt */
    i64 *ihi, *ilo;              /* [S] index of hi[s] / lo[s] - 1 among the points */
    int part;                    /* the intervals partition a range: U = S + 1, ihi[s] = ilo[s] + 1 */
    i64 *elo, *ehi;              /* [LOGS_HI - LOGS_LO + 1] the window thresholds of inv_s = INVS_TAB[logs - LOGS_LO] (mix_freqs) */
    int32_t *sig32;              /* SIG_TAB as int32 (values in [0, 2^16]: half the cache footprint) */
    int32_t kstart[130];         /* first point whose key (sign x bit length) is >= k (lbound_fast) */
    i64 *pairsym;                /* [U] partition: the symbol whose interval is (pt[u - 1], pt[u]] */
    const i64 *exp_tab, *invs_tab, *sig_tab, *log_tab;
    i64 exp_lim, logs_lo, logs_hi, sig_lim, mu_lim, M;
} dl_mix;

void dl_mix_free(dl_mix *m)
{
    if (!m) return;
    free(m->lo); free(m->hi); free(m->wd); free(m->dq); free(m->ihi); free(m->ilo); free(m->pairsym); free(m->elo); free(m->ehi); free(m->sig32); free(m);
}

static int cmp_i64(const void *a, const void *b) { i64 x = *(const i64 *)a, y = *(const i64 *)b; return (x > y) - (x < y); }

static i64 find_pt(const i64 *pts, i64 U, i64 v)
{
    i64 a = 0, b = U - 1;
    while (a < b) { i64 c = (a + b) >> 1; if (pts[c] < v) a = c + 1; else b = c; }
    return a;
}

/* a monotone key of an int64: 64 +- the bit length of |v| (0 .. 128) */
static inline int dkey(i64 v) { return v >= 0 ? 64 + bitlen((u64)v) : 64 - bitlen((u64)0 - (u64)v); }

/* The tables are the shipped feature tables (kept alive by the caller). */
dl_mix *dl_mix_create(i64 S, const i64 *lo, const i64 *hi, const i64 *wd, const i64 *exp_tab, i64 exp_lim, const i64 *invs_tab,
                      i64 logs_lo, i64 logs_hi, const i64 *sig_tab, i64 sig_lim, i64 mu_lim, i64 rans_scale, const i64 *log_tab)
{
    dl_mix *m = (dl_mix *)calloc(1, sizeof(dl_mix));
    if (!m) return 0;
    m->S = S;
    m->lo = (i64 *)malloc(sizeof(i64) * S); m->hi = (i64 *)malloc(sizeof(i64) * S); m->wd = (i64 *)malloc(sizeof(i64) * S);
    m->ihi = (i64 *)malloc(sizeof(i64) * S); m->ilo = (i64 *)malloc(sizeof(i64) * S);
    i64 *pts = (i64 *)malloc(sizeof(i64) * 2 * S);
    m->dq = (i64 *)malloc(sizeof(i64) * 2 * S);
    if (!m->lo || !m->hi || !m->wd || !m->ihi || !m->ilo || !pts || !m->dq) { free(pts); dl_mix_free(m); return 0; }
    memcpy(m->lo, lo, sizeof(i64) * S); memcpy(m->hi, hi, sizeof(i64) * S); memcpy(m->wd, wd, sizeof(i64) * S);
    for (i64 s = 0; s < S; s++) {
        if (wd[s] < 0 || wd[s] > 62 || lo[s] > hi[s]) { free(pts); dl_mix_free(m); return 0; }
        pts[s] = hi[s]; pts[S + s] = lo[s] - 1;
    }
    qsort(pts, (size_t)(2 * S), sizeof(i64), cmp_i64);
    i64 U = 0;
    for (i64 q = 0; q < 2 * S; q++) if (q == 0 || pts[q] != pts[U - 1]) pts[U++] = pts[q];
    m->U = U;
    for (i64 s = 0; s < S; s++) { m->ihi[s] = find_pt(pts, U, hi[s]); m->ilo[s] = find_pt(pts, U, lo[s] - 1); }
    m->pairsym = (i64 *)malloc(sizeof(i64) * (size_t)U);
    if (!m->pairsym) { free(pts); dl_mix_free(m); return 0; }
    m->part = U == S + 1;
    for (i64 u = 0; u < U; u++) m->pairsym[u] = -1;
    for (i64 s = 0; s < S && m->part; s++) {
        if (m->ihi[s] != m->ilo[s] + 1 || m->pairsym[m->ihi[s]] >= 0) m->part = 0;
        else m->pairsym[m->ihi[s]] = s;
    }
    for (i64 u = 0; u < U; u++) m->dq[u] = clipi(wadd(shl(pts[u], 8), 128), -mu_lim, mu_lim);    /* np.clip((pts << 8) + 128, ...) */
    free(pts);
    m->exp_tab = exp_tab; m->invs_tab = invs_tab; m->sig_tab = sig_tab; m->log_tab = log_tab;
    m->exp_lim = exp_lim; m->logs_lo = logs_lo; m->logs_hi = logs_hi; m->sig_lim = sig_lim; m->mu_lim = mu_lim;
    m->M = (i64)1 << rans_scale;
    const i64 nl = logs_hi - logs_lo + 1;
    m->elo = (i64 *)malloc(sizeof(i64) * (size_t)(nl > 0 ? nl : 1)); m->ehi = (i64 *)malloc(sizeof(i64) * (size_t)(nl > 0 ? nl : 1));
    if (!m->elo || !m->ehi || (sig_lim << 16) >= mu_lim) { dl_mix_free(m); return 0; }
    for (int k = 0; k < 130; k++) {
        i64 u = 0;
        while (u < m->U && dkey(m->dq[u]) < k) u++;
        m->kstart[k] = (int32_t)u;
    }
    m->sig32 = (int32_t *)malloc(sizeof(int32_t) * (size_t)(2 * sig_lim + 1));
    if (!m->sig32) { dl_mix_free(m); return 0; }
    for (i64 q = 0; q <= 2 * sig_lim; q++) {
        if (sig_tab[q] < 0 || sig_tab[q] > ((i64)1 << 30)) { dl_mix_free(m); return 0; }
        m->sig32[q] = (int32_t)sig_tab[q];
    }
    for (i64 q = 0; q < nl; q++) {
        const i64 invs = invs_tab[q];
        if (invs <= 0) { m->elo[q] = 0; m->ehi[q] = 0; continue; }
        m->elo[q] = -(((sig_lim - 1) << 16) / invs);                        /* see mix_freqs */
        m->ehi[q] = ((sig_lim << 16) + invs - 1) / invs;
    }
    return m;
}

/* z(u) of one component: clip(clip(d_u - mu, +-MU_LIM) * inv_s >> 16, +-SIG_LIM) (non-decreasing in u) */
static inline i64 zof(const dl_mix *m, i64 u, i64 mu, i64 invs)
{
    i64 z = clipi(wsub(m->dq[u], mu), -m->mu_lim, m->mu_lim);
    z = asr(wmul(z, invs), 16);
    return clipi(z, -m->sig_lim, m->sig_lim);
}

/* first u with dq[u] >= v: every point with a smaller key is below v, so start at the first point of v's key and step
 * over the few points of that key (two per octave for the sign-magnitude alphabet) */
static inline i64 lbound_fast(const dl_mix *m, i64 v)
{
    i64 u = m->kstart[dkey(v)];
    while (u < m->U && m->dq[u] < v) u++;
    return u;
}

/* freqs of one row: f [S] (sum 2^15, every entry >= 1), exactly features.symbol_freqs.  Fv, D: scratch [U + 1].
 * Per component the points split into z <= -SIG_LIM (sigmoid table entry 0), the window, and z >= SIG_LIM (entry
 * 2 SIG_LIM); the window's ends are found on the sorted points: with e = clip(d - mu, +-MU_LIM), inv_s > 0,
 *   z > -L  <=>  e inv_s >= (1 - L) 2^16  <=>  d >= mu + e_lo,  e_lo = -floor((L - 1) 2^16 / inv_s)   (> -MU_LIM)
 *   z >= L  <=>  e inv_s >= L 2^16        <=>  d >= mu + e_hi,  e_hi = ceil(L 2^16 / inv_s)            (<  MU_LIM)
 * (floor(v / 2^16) > a <=> v >= (a + 1) 2^16 for integers; clip(d - mu) is monotone and |e_lo|, e_hi <= L 2^16 <
 * MU_LIM, so the clip never decides); inv_s = 0 gives z = 0 at every point (all in the window).
 * Returns 0, or DL_E_RANGE if logs is outside the table (numpy would index out of range there). */
static inline __attribute__((always_inline)) int mix_freqs_k(const dl_mix *m, const i64 K, const i64 *lw, const i64 *mu, const i64 *ls, i64 *f,
                                                            i64 *Fv, i64 *D)
{
    const i64 S = m->S, U = m->U, L = m->sig_lim;
    i64 mx = lw[0];
    for (i64 k = 1; k < K; k++) if (lw[k] > mx) mx = lw[k];
    i64 e[16], w[16] = {0}, tot = 0, am = 0;
    for (i64 k = 0; k < K; k++) {
        i64 z = clipi(wsub(lw[k], mx), -m->exp_lim, 0);
        e[k] = m->exp_tab[z + m->exp_lim]; tot += e[k];
        if (e[k] > e[am]) am = k;                                            /* np.argmax: the first maximum */
    }
    i64 wsum = 0;
    for (i64 k = 0; k < K; k++) { w[k] = (e[k] << 16) / tot; wsum += w[k]; }
    w[am] += 65536 - wsum;
    const i64 s_lo = m->sig_tab[0], s_hi = m->sig_tab[2 * L];
    const int32_t *sig = m->sig32 + L;
    i64 ua[16], ub[16];
    for (i64 k = 0; k < K; k++) {
        if (ls[k] < m->logs_lo || ls[k] > m->logs_hi) return DL_E_RANGE;
        const i64 q = ls[k] - m->logs_lo, invs = m->invs_tab[q], muk = mu[k];
        if (invs <= 0) { ua[k] = 0; ub[k] = U; }
        else if (muk > ((i64)1 << 61) || muk < -((i64)1 << 61)) {
            i64 a = 0, b = U;                                                /* far-out mean: bisection on z itself */
            while (a < b) { i64 c = (a + b) >> 1; if (zof(m, c, muk, invs) > -L) b = c; else a = c + 1; }
            ua[k] = a; b = U;
            while (a < b) { i64 c = (a + b) >> 1; if (zof(m, c, muk, invs) >= L) b = c; else a = c + 1; }
            ub[k] = a;
        } else {
            ua[k] = lbound_fast(m, muk + m->elo[q]);
            ub[k] = lbound_fast(m, muk + m->ehi[q]);
        }
    }
    const i64 MS = m->M - S;
    if (m->part) {
        /* F changes only on [u0, u1] = [min ua - 1, max ub] (outside every component is saturated on one side), so
         * P = 0 (f = 1) for every symbol outside, and ptot = F[u1] - F[u0] (F is non-decreasing: telescoping) */
        i64 u0 = U, u1 = 0;
        for (i64 k = 0; k < K; k++) { if (ua[k] < u0) u0 = ua[k]; if (ub[k] > u1) u1 = ub[k]; }
        u0 = u0 > 0 ? u0 - 1 : 0; u1 = u1 < U - 1 ? u1 : U - 1;
        if (u1 < u0) u1 = u0;
        const i64 nu = u1 - u0 + 1;
        for (i64 j = 0; j < nu; j++) {                                       /* the saturated parts */
            const i64 u = u0 + j;
            i64 acc = 0;
            for (i64 k = 0; k < K; k++) acc += (u < ua[k] ? s_lo : (u >= ub[k] ? s_hi : 0)) * w[k];
            Fv[j] = acc;
        }
        for (i64 k = 0; k < K; k++) {                                        /* the windows (inside [u0, u1]) */
            const i64 invs = m->invs_tab[ls[k] - m->logs_lo], muk = mu[k], wk = w[k];
            i64 u = ua[k];
#if DL_AVX512
            /* 8 points at a time: the same wrap-around int64 arithmetic (vpsubq / vpmullq / vpsraq), clips, table gather */
            const __m512i vmu = _mm512_set1_epi64(muk), vinv = _mm512_set1_epi64(invs), vw = _mm512_set1_epi64(wk);
            const __m512i vml = _mm512_set1_epi64(-m->mu_lim), vmh = _mm512_set1_epi64(m->mu_lim);
            const __m512i vsl = _mm512_set1_epi64(-L), vsh = _mm512_set1_epi64(L);
            for (; u + 8 <= ub[k]; u += 8) {
                __m512i z = _mm512_sub_epi64(_mm512_loadu_si512((const void *)(m->dq + u)), vmu);
                z = _mm512_min_epi64(_mm512_max_epi64(z, vml), vmh);
                z = _mm512_srai_epi64(_mm512_mullo_epi64(z, vinv), 16);
                z = _mm512_min_epi64(_mm512_max_epi64(z, vsl), vsh);
                const __m512i sv = _mm512_cvtepi32_epi64(_mm512_i64gather_epi32(z, (const void *)sig, 4));
                i64 *fp = Fv + (u - u0);
                _mm512_storeu_si512((void *)fp, _mm512_add_epi64(_mm512_loadu_si512((const void *)fp), _mm512_mullo_epi64(sv, vw)));
            }
#endif
            for (; u < ub[k]; u++) Fv[u - u0] += (i64)sig[zof(m, u, muk, invs)] * wk;
        }
        for (i64 j = 0; j < nu; j++) Fv[j] >>= 16;
        const i64 ptot = Fv[nu - 1] - Fv[0];
        const i64 den = ptot > 1 ? ptot : 1;
        const double inv = 1.0 / (double)den;
        for (i64 s = 0; s < S; s++) f[s] = 1;
        i64 extra = 0;
        for (i64 j = 1; j < nu; j++) {
            const i64 a = (Fv[j] - Fv[j - 1]) * MS;
            i64 q = (i64)((double)a * inv);
            q -= q * den > a;
            q += (q + 1) * den <= a;
            f[m->pairsym[u0 + j]] = 1 + q; extra += q;
        }
        f[0] += m->M - (S + extra);
        return 0;
    }
    memset(Fv, 0, sizeof(i64) * (size_t)U); memset(D, 0, sizeof(i64) * (size_t)(U + 1));
    for (i64 k = 0; k < K; k++) {
        const i64 invs = m->invs_tab[ls[k] - m->logs_lo], muk = mu[k], wk = w[k];
        for (i64 u = ua[k]; u < ub[k]; u++) Fv[u] += (i64)sig[zof(m, u, muk, invs)] * wk;
        D[0] += s_lo * wk; D[ua[k]] -= s_lo * wk;                           /* u < ua: saturated low */
        D[ub[k]] += s_hi * wk;                                               /* u >= ub: saturated high */
    }
    i64 run = 0;
    for (i64 u = 0; u < U; u++) { run += D[u]; Fv[u] = (Fv[u] + run) >> 16; }
    i64 ptot = 0;
    for (i64 s = 0; s < S; s++) { i64 P = Fv[m->ihi[s]] - Fv[m->ilo[s]]; P = P < 0 ? 0 : P; f[s] = P; ptot += P; }
    const i64 den = ptot > 1 ? ptot : 1;
    /* f = 1 + floor(P (M - S) / den), 0 <= P (M - S) < 2^31, den <= 2^17: q = trunc(a * (1 / den)) is within one of
     * the floor (relative error < 2^-51 of a < 2^31), one integer correction each way makes it exact */
    const double inv = 1.0 / (double)den;
    i64 fsum = 0;
    for (i64 s = 0; s < S; s++) {
        const i64 a = f[s] * MS;
        i64 q = (i64)((double)a * inv);
        q -= q * den > a;
        q += (q + 1) * den <= a;
        f[s] = 1 + q; fsum += f[s];
    }
    f[0] += m->M - fsum;
    return 0;
}

/* K = 3 (every shipped model) gets its own unrolled copy */
static int mix_freqs(const dl_mix *m, i64 K, const i64 *lw, const i64 *mu, const i64 *ls, i64 *f, i64 *Fv, i64 *D)
{
    if (K == 3) return mix_freqs_k(m, 3, lw, mu, ls, f, Fv, D);
    return mix_freqs_k(m, K, lw, mu, ls, f, Fv, D);
}

/* ================================================================================================================ */
/* the features (features.CurveState) -- one curve's register state                                                */
/* ================================================================================================================ */
static inline i64 ilog2_q8(const i64 *log_tab, i64 v)
{
    if (v < 1) v = 1;
    int e = bitlen((u64)v) - 1;
    i64 f = (e >= 8 ? (v >> (e - 8)) : (i64)((u64)v << (8 - e))) - 256;
    if (f < 0) f = 0;
    if (f > 255) f = 255;
    return (i64)e * 256 + log_tab[f];
}
static inline i64 Lq(const i64 *lt, i64 v) { return ilog2_q8(lt, v + 1); }
static inline i64 slog2q(const i64 *lt, i64 x) { return x > 0 ? Lq(lt, x) : (x < 0 ? -Lq(lt, -x) : 0); }

/* ================================================================================================================ */
/* the lockstep residual loop                                                                                       */
/* ================================================================================================================ */
enum { P_D1 = 0, P_D2 = 1, P_D2T = 2, P_NLMS = 3 };

typedef struct {
    i64 lh[HH], ew[NTAU], ld[3], la[2], d0, xprev;      /* CurveState registers (transformed history, EWMAs, differences) */
    i64 x, xp, hist[8], w[8];                           /* predict.IncPred */
    int ptype;
} col_state;

/* C coded columns in coder order:
 *   n [C] residuals per column, pid [C], kf [C], base [C, 36] (CurveState.base with the step column), x0 [C],
 *   goff [C] offsets of each keyframe column's n gaps in gaps (all-keys columns: -1),
 *   xoff [C] offsets of each column's n + 1 decoded integers in X (X[xoff] = x0).
 * K = mixture components of the model (out rows are 3 K wide).  NLMS constants as predict.py.
 * Returns 0, a negative error, or DL_E_MODEL + rc (rc = the hook's nonzero return). */
i64 dl_lockstep(dl_dec *d, const dl_mix *m, i64 K, dl_step_fn fn, void *ctx,
                i64 C, const i64 *n, const i64 *pid, const i64 *kf, const i64 *base, const i64 *goff, const i64 *gaps,
                const i64 *x0, const i64 *xoff, i64 *X,
                i64 order, i64 nlms_sh, i64 nlms_mu, i64 clamp, i64 wmax)
{
    if (K < 1 || K > 16 || order < 0 || order > 8 || C <= 0) return DL_E_ARG;
    const i64 *lt = m->log_tab;
    i64 nmax = 0;
    for (i64 c = 0; c < C; c++) { if (n[c] > nmax) nmax = n[c]; if (n[c] < 0) return DL_E_ARG; }
    const i64 S = m->S, U = m->U;
    col_state *cs = (col_state *)calloc((size_t)C, sizeof(col_state));
    i64 *act = (i64 *)malloc(sizeof(i64) * (size_t)C);
    i64 *feat = (i64 *)malloc(sizeof(i64) * (size_t)C * NF);
    i64 *out = (i64 *)malloc(sizeof(i64) * (size_t)C * 3 * K);
    i64 *fb = (i64 *)malloc(sizeof(i64) * (size_t)(S * C));
    i64 *Fv = (i64 *)malloc(sizeof(i64) * (size_t)(2 * U + 1));
    i64 *lgoff = (i64 *)malloc(sizeof(i64) * (size_t)C);
    i64 *lg = 0;
    i64 ng = 0;
    for (i64 c = 0; c < C; c++) if (kf[c] && goff[c] >= 0) ng += n[c];
    lg = (i64 *)malloc(sizeof(i64) * (size_t)(ng > 0 ? ng : 1));
    int rc = 0;
    if (!cs || !act || !feat || !out || !fb || !Fv || !lgoff || !lg) { rc = DL_E_ALLOC; goto done; }
    {
        /* CurveState.lg = L(max(g, 1) - 1) per keyframe column residual (0 for all-keys columns) */
        i64 o = 0;
        for (i64 c = 0; c < C; c++) {
            lgoff[c] = -1;
            if (kf[c] && goff[c] >= 0) {
                lgoff[c] = o;
                for (i64 j = 0; j < n[c]; j++) { i64 g = gaps[goff[c] + j]; lg[o + j] = Lq(lt, (g > 1 ? g : 1) - 1); }
                o += n[c];
            }
        }
    }
    const i64 half = (i64)1 << (nlms_sh - 1);
    const int sh = (int)(nlms_sh - nlms_mu);
    i64 n_act = 0;
    for (i64 c = 0; c < C; c++) {
        col_state *s = cs + c;
        for (int k = 0; k < NTAU; k++) s->ew[k] = 256;                           /* EM0 */
        s->xprev = x0[c]; s->x = x0[c]; s->xp = x0[c];
        if (order > 0) s->w[0] = (i64)1 << nlms_sh;                             /* NLMS_INIT = 'd2' */
        const i64 p = pid[c];
        s->ptype = p == 1 ? P_D2 : ((kf[c] && p == 2) ? P_D2T : (((!kf[c] && p == 2) || (kf[c] && p == 3)) ? P_NLMS : P_D1));
        X[xoff[c]] = x0[c];
        if (n[c] > 0) act[n_act++] = c;
    }
    const i64 Mmask = m->M - 1;
    for (i64 t = 0; t < nmax; t++) {
        /* the active columns (ascending; a column is active exactly for t < n) */
        i64 k2 = 0;
        for (i64 i = 0; i < n_act; i++) if (n[act[i]] > t) act[k2++] = act[i];
        n_act = k2;
        /* features (CurveState.features) */
        const i64 Lt = Lq(lt, t);
        for (i64 i = 0; i < n_act; i++) {
            const i64 c = act[i];
            const col_state *s = cs + c;
            i64 *F = feat + i * NF;
            const i64 *bs = base + c * NF;
            for (int k = 0; k < NF; k++) F[k] = bs[k];
            for (int k = 0; k < HH; k++) F[k] = s->lh[k];
            for (int k = 0; k < NTAU; k++) F[HH + k] = ilog2_q8(lt, 256 + s->ew[k]) - 8 * 256;
            int q = HH + NTAU;
            F[q] = s->ld[0]; F[q + 1] = s->ld[1]; F[q + 2] = s->ld[2]; F[q + 3] = s->la[0]; F[q + 4] = s->la[1];
            q += 5;
            if (lgoff[c] >= 0) {
                /* CurveState indexes the batch's gap matrix [nmax, C] (zero past this column's end) */
                const i64 *l = lg + lgoff[c];
                const i64 kb = t < nmax - 1 ? t : nmax - 1, ka = (t + 1) < nmax - 1 ? (t + 1) : nmax - 1;
                const i64 gb = kb < n[c] ? l[kb] : 0, ga = ka < n[c] ? l[ka] : 0;
                const i64 gp = t > 0 ? l[t - 1] : gb;
                F[q] = gb; F[q + 1] = ga; F[q + 2] = gb - gp;
            } else {
                F[q] = 0; F[q + 1] = 0; F[q + 2] = 0;
            }
            F[q + 3 + 13] = Lt;
        }
        /* the model */
        int mrc = fn(ctx, t, n_act, act, feat, out);
        if (mrc) { rc = DL_E_MODEL + mrc; goto done; }
        /* the frequency tables of the step's rows (one pass: the tables stay in cache), then per column (coder order):
         * symbol, raw low bits, residual, predictor, state update */
        for (i64 i = 0; i < n_act; i++) {
            const i64 *o = out + i * 3 * K;
            rc = mix_freqs(m, K, o, o + K, o + 2 * K, fb + i * S, Fv, Fv + U);
            if (rc) goto done;
        }
        for (i64 i = 0; i < n_act; i++) {
            const i64 c = act[i];
            col_state *s = cs + c;
            const i64 *f = fb + i * S;
            const i64 slot = (i64)(d->x & (u64)Mmask);
            i64 sym = 0, cum = 0;
            while (cum + f[sym] <= slot) { cum += f[sym]; sym++; }             /* searchsorted(cumsum(f), slot, 'right') */
            u64 x = (u64)f[sym] * (d->x >> d->SB) + (u64)(slot - cum);
            while (x < (u64)d->L) {
                if (d->rp >= d->rn) { rc = DL_E_EOF_RANS; goto done; }
                x = (x << 8) | d->rd[d->rp++];
            }
            d->x = x;
            const i64 w = m->wd[sym];
            const i64 low = w ? (i64)br_read(d, (int)w) : 0;
            if (d->err) { rc = d->err; goto done; }
            const i64 a = m->lo[sym];
            const i64 r = a >= 0 ? wadd(a, low) : wsub(m->hi[sym], low);    /* features.symbol_value */
            /* IncPred.predict */
            i64 P = s->x;
            if (s->ptype == P_NLMS) {
                i64 acc = 0;
                for (i64 k = 0; k < order; k++) acc = wadd(acc, wmul(s->w[k], s->hist[k]));
                P = wadd(s->x, asr(wadd(acc, half), (int)nlms_sh));
            } else if (t >= 1 && s->ptype != P_D1) {
                const i64 dd = wsub(s->x, s->xp);
                if (s->ptype == P_D2) P = wadd(s->x, dd);
                else {
                    /* numpy: ratio = float64(g[t]) / float64(max(g[t-1], 1)); x + rint(float64(d) * ratio) -- two IEEE
                     * operations and a round-half-even, the same in C (no contraction possible: nothing is added) */
                    const i64 *g = gaps + goff[c];
                    const i64 gm = g[t - 1] > 1 ? g[t - 1] : 1;
                    const double ratio = (double)g[t] / (double)gm;
                    P = wadd(s->x, (i64)rint((double)dd * ratio));
                }
            }
            const i64 xn = wadd(P, r);
            X[xoff[c] + t + 1] = xn;
            /* IncPred.update (only NLMS columns read w / hist) */
            if (s->ptype == P_NLMS) {
                const i64 e = clipi(wsub(xn, P), -clamp, clamp);
                const i64 dn = wsub(xn, s->x);
                i64 hc[8], norm = 0;
                for (i64 k = 0; k < order; k++) { hc[k] = clipi(s->hist[k], -clamp, clamp); norm = wadd(norm, wmul(hc[k], hc[k])); }
                norm = wadd(norm, order);
                for (i64 k = 0; k < order; k++) s->w[k] = clipi(wadd(s->w[k], floordiv(shl(wmul(e, hc[k]), sh), norm)), -wmax, wmax);
                for (i64 k = order - 1; k > 0; k--) s->hist[k] = s->hist[k - 1];
                if (order > 0) s->hist[0] = dn;
            }
            s->xp = s->x; s->x = xn;
            /* CurveState.update(r, x_new) */
            const i64 u = iabs(r);
            for (int k = HH - 1; k > 0; k--) s->lh[k] = s->lh[k - 1];
            s->lh[0] = slog2q(lt, r);
            for (int k = 0; k < NTAU; k++) s->ew[k] = wadd(s->ew[k], asr(wsub(shl(u, 8), s->ew[k]), k + 1));
            const i64 dn2 = wsub(xn, s->xprev);
            s->la[1] = s->la[0]; s->la[0] = slog2q(lt, wsub(dn2, s->d0));
            s->ld[2] = s->ld[1]; s->ld[1] = s->ld[0]; s->ld[0] = slog2q(lt, dn2);
            s->d0 = dn2; s->xprev = xn;
        }
    }
done:
    free(cs); free(act); free(feat); free(out); free(fb); free(Fv); free(lgoff); free(lg);
    if (!rc && d->err) rc = d->err;
    return rc;
}

/* ================================================================================================================ */
/* the whole blob: header -> columns -> lockstep -> interpolation (dl_v2)                                          */
/* ================================================================================================================ */
typedef struct {
    i64 F, B, ntr, nrot;
    i64 *bone, *I3, *x0, *mode, *pid, *cidx, *crot, *ctr;   /* [ntr], [3 ntr] x 2, [ntr], [3 ntr], [B], [3 B] x 2 */
    i64 *koff, *kn;                                         /* [3 ntr] key set of every track component in K (-1: all frames) */
    i64 *K; i64 Kn, Kcap;
    i64 C, ng, nX;                                          /* columns */
    i64 *cn, *cpid, *ckf, *cgoff, *cx0, *cxoff, *base, *gaps, *X, *colof;   /* colof [3 ntr]: the column of a component or -1 */
} dl_v2;

void dl_v2_free(dl_v2 *v)
{
    if (!v) return;
    free(v->bone); free(v->I3); free(v->x0); free(v->mode); free(v->pid); free(v->cidx); free(v->crot); free(v->ctr);
    free(v->koff); free(v->kn); free(v->K);
    free(v->cn); free(v->cpid); free(v->ckf); free(v->cgoff); free(v->cx0); free(v->cxoff); free(v->base); free(v->gaps); free(v->X);
    free(v->colof); free(v);
}

dl_v2 *dl_v2_new(void) { return (dl_v2 *)calloc(1, sizeof(dl_v2)); }

static i64 *kpush(dl_v2 *v, i64 n)
{
    if (v->Kn + n > v->Kcap) {
        i64 cap = v->Kcap ? v->Kcap : 1024;
        while (cap < v->Kn + n) cap *= 2;
        i64 *p = (i64 *)realloc(v->K, sizeof(i64) * (size_t)cap);
        if (!p) return 0;
        v->K = p; v->Kcap = cap;
    }
    i64 *r = v->K + v->Kn; v->Kn += n;
    return r;
}

/* cfg (int64): see decloop.py _V2CFG.  rts / tts: the B type symbols (no pointer types: the caller checked).
 * Mirrors _V2Pack.decode from `stepd = {}` to the key sets, symbol for symbol. */
enum { G_TRA, G_TTA, G_TRC, G_TRR, G_TTC, G_FROT, G_FTR, G_KEG, G_PC, G_CTPCD, G_CFIRST, G_CEGK, G_CX0R, G_CX0T, G_CMODE, G_CPIDA,
       G_CPIDK, G_USEIDS, G_MALL, G_MKF, G_MCK, G_MSTAT, G_CGAP, G_NGAP, G_GCAP, G_ESC, G_FB, G_NCFG };

i64 dl_v2_header(dl_v2 *v, dl_dec *d, i64 F, i64 B, const i64 *rts, const i64 *tts, const i64 *cfg)
{
    v->F = F; v->B = B;
    i64 ntr = 0, nrot = 0;
    for (i64 b = 0; b < B; b++) if (rts[b] == cfg[G_TRA]) nrot++;
    ntr = nrot;
    for (i64 b = 0; b < B; b++) if (tts[b] == cfg[G_TTA]) ntr++;
    v->ntr = ntr; v->nrot = nrot;
    const size_t n1 = (size_t)(ntr > 0 ? ntr : 1), n3 = 3 * n1, b3 = (size_t)(3 * (B > 0 ? B : 1));
    v->bone = (i64 *)calloc(n1, 8); v->I3 = (i64 *)calloc(n3, 8); v->x0 = (i64 *)calloc(n3, 8); v->mode = (i64 *)calloc(n1, 8);
    v->pid = (i64 *)calloc(n3, 8); v->cidx = (i64 *)calloc((size_t)(B > 0 ? B : 1), 8); v->crot = (i64 *)calloc(b3, 8);
    v->ctr = (i64 *)calloc(b3, 8); v->koff = (i64 *)calloc(n3, 8); v->kn = (i64 *)calloc(n3, 8);
    if (!v->bone || !v->I3 || !v->x0 || !v->mode || !v->pid || !v->cidx || !v->crot || !v->ctr || !v->koff || !v->kn) return DL_E_ALLOC;
    i64 t = 0;
    for (i64 b = 0; b < B; b++) if (rts[b] == cfg[G_TRA]) v->bone[t++] = b;
    for (i64 b = 0; b < B; b++) if (tts[b] == cfg[G_TTA]) v->bone[t++] = b;
    /* grid indices: rot tracks, then trans tracks, each kind delta-coded from its first value */
    for (int kind = 0; kind < 2; kind++) {
        i64 prev = kind ? cfg[G_FTR] : cfg[G_FROT];
        const i64 t0 = kind ? nrot : 0, t1 = kind ? ntr : nrot;
        for (i64 i = t0; i < t1; i++) {
            prev = wadd(prev, unzigzag(br_expg(d, (int)cfg[G_KEG])));
            v->I3[3 * i] = prev; v->I3[3 * i + 1] = prev; v->I3[3 * i + 2] = prev;
            if (cfg[G_PC]) {
                const i64 a = get_mag(d, cfg[G_CTPCD]), c = get_mag(d, cfg[G_CTPCD]);
                v->I3[3 * i + 1] = wadd(prev, a); v->I3[3 * i + 2] = wadd(prev, c);
            }
        }
    }
    { i64 prev = cfg[G_CFIRST];
      for (i64 b = 0; b < B; b++) if (rts[b] == cfg[G_TRC] || rts[b] == cfg[G_TRR]) { prev = wadd(prev, unzigzag(br_expg(d, (int)cfg[G_CEGK]))); v->cidx[b] = prev; } }
    for (i64 b = 0; b < B; b++) if (rts[b] == cfg[G_TRC]) for (int c = 0; c < 3; c++) v->crot[3 * b + c] = get_mag(d, cfg[G_CX0R]);
    for (i64 b = 0; b < B; b++) if (tts[b] == cfg[G_TTC]) for (int c = 0; c < 3; c++) v->ctr[3 * b + c] = get_mag(d, cfg[G_CX0T]);
    for (i64 i = 0; i < ntr; i++) for (int c = 0; c < 3; c++) v->x0[3 * i + c] = get_mag(d, i < nrot ? cfg[G_CX0R] : cfg[G_CX0T]);
    for (i64 i = 0; i < ntr; i++) {
        const i64 mo = hdr_decode(d, cfg[G_CMODE]);
        if (mo != cfg[G_MALL] && mo != cfg[G_MKF] && mo != cfg[G_MCK] && mo != cfg[G_MSTAT]) { if (!d->err) d->err = DL_E_RANGE; }
        v->mode[i] = mo;
    }
    if (d->err) return d->err;
    for (i64 i = 0; i < ntr; i++) if (v->mode[i] == cfg[G_MALL]) for (int c = 0; c < 3; c++) v->pid[3 * i + c] = cfg[G_USEIDS] ? hdr_decode(d, cfg[G_CPIDA]) : 2;
    for (i64 i = 0; i < ntr; i++) {
        if (v->mode[i] == cfg[G_MKF]) { const i64 p = hdr_decode(d, cfg[G_CPIDK]); v->pid[3 * i] = v->pid[3 * i + 1] = v->pid[3 * i + 2] = p; }
        else if (v->mode[i] == cfg[G_MCK]) for (int c = 0; c < 3; c++) v->pid[3 * i + c] = hdr_decode(d, cfg[G_CPIDK]);
    }
    if (d->err) return d->err;
    /* key sets */
    i64 *U = (i64 *)malloc(sizeof(i64) * (size_t)(F + 2));
    uint8_t *pres = (uint8_t *)malloc((size_t)(3 * (F + 2)));
    if (!U || !pres) { free(U); free(pres); return DL_E_ALLOC; }
    i64 rc = 0;
    for (i64 i = 0; i < ntr && !rc; i++) {
        const i64 mo = v->mode[i];
        if (mo == cfg[G_MKF] || mo == cfg[G_MCK]) {
            const int isck = mo == cfg[G_MCK];
            const i64 kbase = cfg[G_CGAP] + (i < nrot ? 0 : cfg[G_NGAP]);
            const i64 nU = dl_keys(d, F, kbase, cfg[G_GCAP], cfg[G_ESC], isck, cfg[G_FB], F + 2, U, pres);
            if (nU < 0) { rc = nU; break; }
            if (!isck) {
                i64 *k = kpush(v, nU);
                if (!k) { rc = DL_E_ALLOC; break; }
                memcpy(k, U, sizeof(i64) * (size_t)nU);
                for (int c = 0; c < 3; c++) { v->koff[3 * i + c] = k - v->K; v->kn[3 * i + c] = nU; }
            } else {
                for (int c = 0; c < 3; c++) {
                    i64 m = 0;
                    for (i64 j = 0; j < nU; j++) m += pres[3 * j + c];
                    i64 *k = kpush(v, m);
                    if (!k) { rc = DL_E_ALLOC; break; }
                    m = 0;
                    for (i64 j = 0; j < nU; j++) if (pres[3 * j + c]) k[m++] = U[j];
                    v->koff[3 * i + c] = k - v->K; v->kn[3 * i + c] = m;
                }
            }
        } else if (mo == cfg[G_MALL]) {
            for (int c = 0; c < 3; c++) { v->koff[3 * i + c] = -1; v->kn[3 * i + c] = F; }
        } else {                                                           /* static: [0, F - 1] (or [0]) */
            i64 *k = kpush(v, F >= 2 ? 2 : 1);
            if (!k) { rc = DL_E_ALLOC; break; }
            k[0] = 0; if (F >= 2) k[1] = F - 1;
            for (int c = 0; c < 3; c++) { v->koff[3 * i + c] = k - v->K; v->kn[3 * i + c] = F >= 2 ? 2 : 1; }
        }
    }
    free(U); free(pres);
    return rc ? rc : d->err;
}

/* sizes: ntr, nrot, C, Kn, nX */
void dl_v2_sizes(const dl_v2 *v, i64 *out) { out[0] = v->ntr; out[1] = v->nrot; out[2] = v->C; out[3] = v->Kn; out[4] = v->nX; }

/* copy the header arrays out: bone [ntr], I3 [3 ntr], x0 [3 ntr], mode [ntr], pid [3 ntr], cidx [B], crot [3 B], ctr [3 B] */
void dl_v2_arrays(const dl_v2 *v, i64 *bone, i64 *I3, i64 *x0, i64 *mode, i64 *pid, i64 *cidx, i64 *crot, i64 *ctr)
{
    const size_t n = (size_t)v->ntr, B = (size_t)v->B;
    memcpy(bone, v->bone, 8 * n); memcpy(I3, v->I3, 24 * n); memcpy(x0, v->x0, 24 * n); memcpy(mode, v->mode, 8 * n);
    memcpy(pid, v->pid, 24 * n); memcpy(cidx, v->cidx, 8 * B); memcpy(crot, v->crot, 24 * B); memcpy(ctr, v->ctr, 24 * B);
}

/* the coded columns (_V2Pack.decode's `cols`: all-keys tracks, then keyframe / per-component tracks, 3 components each)
 * with CurveState's static features.  dcb [B] depth class per bone, G grid steps per octave, fps_i = round(fps), p_idx,
 * step_col = the model's step column (log2 step in Q8: floor(256 idx / G)).  Returns C. */
i64 dl_v2_columns(dl_v2 *v, const i64 *dcb, i64 G, i64 fps_i, i64 p_idx, i64 step_col, const i64 *log_tab, const i64 *cfg)
{
    const i64 ntr = v->ntr;
    v->colof = (i64 *)malloc(sizeof(i64) * (size_t)(3 * (ntr > 0 ? ntr : 1)));
    if (!v->colof) return DL_E_ALLOC;
    for (i64 j = 0; j < 3 * ntr; j++) v->colof[j] = -1;
    i64 C = 0, ng = 0, nX = 0;
    for (int pass = 0; pass < 2; pass++) {                              /* count */
        for (i64 i = 0; i < ntr; i++) {
            const i64 mo = v->mode[i];
            if ((pass == 0) != (mo == cfg[G_MALL])) continue;
            if (mo != cfg[G_MALL] && mo != cfg[G_MKF] && mo != cfg[G_MCK]) continue;
            for (int c = 0; c < 3; c++) {
                const i64 nk = v->kn[3 * i + c];
                if (nk > 1) { C++; nX += nk; if (mo != cfg[G_MALL]) ng += nk - 1; }
            }
        }
    }
    v->C = C; v->ng = ng; v->nX = nX;
    const size_t c1 = (size_t)(C > 0 ? C : 1);
    v->cn = (i64 *)malloc(8 * c1); v->cpid = (i64 *)malloc(8 * c1); v->ckf = (i64 *)malloc(8 * c1); v->cgoff = (i64 *)malloc(8 * c1);
    v->cx0 = (i64 *)malloc(8 * c1); v->cxoff = (i64 *)malloc(8 * c1); v->base = (i64 *)calloc(c1 * NF, 8);
    v->gaps = (i64 *)malloc(8 * (size_t)(ng > 0 ? ng : 1)); v->X = (i64 *)malloc(8 * (size_t)(nX > 0 ? nX : 1));
    if (!v->cn || !v->cpid || !v->ckf || !v->cgoff || !v->cx0 || !v->cxoff || !v->base || !v->gaps || !v->X) return DL_E_ALLOC;
    const i64 lfps = ilog2_q8(log_tab, fps_i);                          /* L(fps_i - 1) */
    i64 j = 0, go = 0, xo = 0;
    for (int pass = 0; pass < 2; pass++) {
        for (i64 i = 0; i < ntr; i++) {
            const i64 mo = v->mode[i];
            if ((pass == 0) != (mo == cfg[G_MALL])) continue;
            if (mo != cfg[G_MALL] && mo != cfg[G_MKF] && mo != cfg[G_MCK]) continue;
            const int kf = mo != cfg[G_MALL];
            for (int c = 0; c < 3; c++) {
                const i64 nk = v->kn[3 * i + c];
                if (nk <= 1) continue;
                const i64 n = nk - 1, pid = v->pid[3 * i + c];
                v->colof[3 * i + c] = j;
                v->cn[j] = n; v->cpid[j] = pid; v->ckf[j] = kf; v->cx0[j] = v->x0[3 * i + c]; v->cxoff[j] = xo; xo += nk;
                if (kf) {
                    const i64 *k = v->K + v->koff[3 * i + c];
                    for (i64 q = 0; q < n; q++) v->gaps[go + q] = k[q + 1] - k[q];
                    v->cgoff[j] = go; go += n;
                } else v->cgoff[j] = -1;
                i64 *bs = v->base + j * NF;                                  /* CurveState.__init__ + the step column */
                const i64 col = HH + NTAU + 5 + 3;
                const i64 dc = dcb[v->bone[i]];
                if (dc < 0 || dc > 3) return DL_E_RANGE;
                bs[col + dc] = 256;
                bs[col + 4] = i < v->nrot ? 0 : 256;
                bs[col + 5] = 256 * kf;
                bs[col + 6 + clipi(pid, 0, 3)] = 256;
                bs[col + 11] = lfps;
                bs[col + 12] = 256 * p_idx;
                bs[col + 14] = ilog2_q8(log_tab, n);                        /* L(n - 1) */
                bs[step_col] = floordiv(wmul(256, v->I3[3 * i + c]), G);    /* _step_q8 */
                j++;
            }
        }
    }
    return C;
}
/* the lockstep residual loop over the columns (dl_lockstep) -> X */
i64 dl_v2_lockstep(dl_v2 *v, dl_dec *d, const dl_mix *m, i64 K, dl_step_fn fn, void *ctx, i64 order, i64 nlms_sh, i64 nlms_mu, i64 clamp,
                   i64 wmax)
{
    if (v->C <= 0) return 0;
    return dl_lockstep(d, m, K, fn, ctx, v->C, v->cn, v->cpid, v->ckf, v->base, v->cgoff, v->gaps, v->cx0, v->cxoff, v->X,
                       order, nlms_sh, nlms_mu, clamp, wmax);
}

/* the integers of track i component c (the decoder's Xd) and its key set: returns nk, fills keys / ints (cap >= F) */
static i64 v2_comp(const dl_v2 *v, i64 i, int c, const i64 **keys, const i64 **ints, i64 *tmp, const i64 *cfg)
{
    const i64 nk = v->kn[3 * i + c], col = v->colof[3 * i + c], mo = v->mode[i];
    *keys = v->koff[3 * i + c] >= 0 ? v->K + v->koff[3 * i + c] : 0;           /* 0: all frames 0 .. F - 1 */
    if (col >= 0) *ints = v->X + v->cxoff[col];
    else {                                                                   /* a single key, or a static track's x0 */
        const i64 x0 = v->x0[3 * i + c];
        tmp[0] = x0; tmp[1] = x0;
        *ints = tmp;
        if (mo != cfg[G_MSTAT] && nk != 1) return -1;
    }
    return nk;
}

/* keys.interp_vectors on one component (fastkern.c interp_vectors with c = 1, the same operation order;
 * compiled without FP contraction) */
static void interp1(i64 nk, const i64 *K, const double *vk, i64 F, int cubic, double *m, double *out, i64 os)
{
    if (nk == F) { for (i64 f = 0; f < F; f++) out[f * os] = vk[f]; return; }
    const int use_cubic = cubic && nk >= 3;
    if (use_cubic) {
        for (i64 k = 1; k < nk - 1; k++) m[k] = (vk[k + 1] - vk[k - 1]) / ((double)K[k + 1] - (double)K[k - 1]);
        m[0] = (vk[1] - vk[0]) / ((double)K[1] - (double)K[0]);
        m[nk - 1] = (vk[nk - 1] - vk[nk - 2]) / ((double)K[nk - 1] - (double)K[nk - 2]);
    }
    i64 j = 0;
    for (i64 f = 0; f < F; f++) {
        while (j + 1 < nk && K[j + 1] <= f) j++;
        i64 jj = j < 0 ? 0 : j;
        if (jj > nk - 2) jj = nk - 2;
        const double t = (double)(f - K[jj]) / (double)(K[jj + 1] - K[jj]);
        const double a = vk[jj], b = vk[jj + 1];
        if (use_cubic) {
            const double h = (double)K[jj + 1] - (double)K[jj];
            const double t2 = t * t, t3 = t2 * t;
            const double c0 = 2 * t3 - 3 * t2 + 1, c1 = t3 - 2 * t2 + t, c2 = -2 * t3 + 3 * t2, c3 = t3 - t2;
            out[f * os] = c0 * a + c1 * h * m[jj] + c2 * b + c3 * h * m[jj + 1];
        } else {
            out[f * os] = a + t * (b - a);
        }
    }
    for (i64 k = 0; k < nk; k++) out[K[k] * os] = vk[k];
}

/* tracks t0 .. t1 - 1: dequantize (float64 product, rounded to float32) + interpolation -> V [t1 - t0, F, 3] float64
 * (keys.reconstruct / keys.reconstruct_c before exp_map / the float32 cast).  step [3 ntr] float32. */
/* the three components of a key set shared by all three (fastkern.c interp_vectors: one segment search and one set
 * of Hermite weights per frame; every output is the same expression as interp1's) */
static void interp3(i64 nk, const i64 *K, const double *vk, i64 F, int cubic, double *m, double *out)
{
    if (nk == F) { for (i64 q = 0; q < 3 * F; q++) out[q] = vk[q]; return; }
    const int use_cubic = cubic && nk >= 3;
    if (use_cubic) {
        for (i64 k = 1; k < nk - 1; k++) {
            const double dt = (double)K[k + 1] - (double)K[k - 1];
            for (int c = 0; c < 3; c++) m[k * 3 + c] = (vk[(k + 1) * 3 + c] - vk[(k - 1) * 3 + c]) / dt;
        }
        const double d0 = (double)K[1] - (double)K[0], d1 = (double)K[nk - 1] - (double)K[nk - 2];
        for (int c = 0; c < 3; c++) {
            m[c] = (vk[3 + c] - vk[c]) / d0;
            m[(nk - 1) * 3 + c] = (vk[(nk - 1) * 3 + c] - vk[(nk - 2) * 3 + c]) / d1;
        }
    }
    i64 j = 0;
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

i64 dl_v2_interp(const dl_v2 *v, i64 t0, i64 t1, i64 cubic, const float *step, const i64 *cfg, double *V)
{
    const i64 F = v->F;
    double *vk = (double *)malloc(sizeof(double) * (size_t)(3 * F + 6)), *m = (double *)malloc(sizeof(double) * (size_t)(3 * F + 6));
    if (!vk || !m) { free(vk); free(m); return DL_E_ALLOC; }
    i64 rc = 0;
    for (i64 i = t0; i < t1 && !rc; i++) {
        double *out = V + (i - t0) * F * 3;
        const int shared = v->koff[3 * i] == v->koff[3 * i + 1] && v->koff[3 * i] == v->koff[3 * i + 2] &&
                           v->kn[3 * i] == v->kn[3 * i + 1] && v->kn[3 * i] == v->kn[3 * i + 2];
        if (shared) {                                                        /* keys.reconstruct */
            const i64 *k = 0, *xc[3]; i64 tmp[3][2], nk = 0;
            for (int c = 0; c < 3; c++) {
                const i64 *kc;
                nk = v2_comp(v, i, c, &kc, &xc[c], tmp[c], cfg);
                if (c == 0) k = kc;
            }
            if (nk < 1 || nk > F || (!k && nk != F) || (k && (k[0] != 0 || k[nk - 1] != F - 1))) { rc = DL_E_RANGE; break; }
            for (int c = 0; c < 3; c++) {
                const double st = (double)step[3 * i + c];
                for (i64 q = 0; q < nk; q++) vk[3 * q + c] = (double)(float)((double)xc[c][q] * st);   /* quantize.dequantize */
            }
            interp3(nk, k, vk, F, (int)cubic, m, out);
            continue;
        }
        for (int c = 0; c < 3; c++) {                                        /* keys.reconstruct_c */
            const i64 *k, *x; i64 tmp[2];
            const i64 nk = v2_comp(v, i, c, &k, &x, tmp, cfg);
            if (nk < 1 || nk > F || (!k && nk != F)) { rc = DL_E_RANGE; break; }
            const double st = (double)step[3 * i + c];
            for (i64 q = 0; q < nk; q++) vk[q] = (double)(float)((double)x[q] * st);     /* quantize.dequantize */
            if (k && (k[0] != 0 || k[nk - 1] != F - 1)) { rc = DL_E_RANGE; break; }
            interp1(nk, k, vk, F, (int)cubic, m, out + c, 3);
        }
    }
    free(vk); free(m);
    return rc;
}

/* rotation tracks t0 .. t1 - 1 into the clip loc [F, B, 7] float32: common.rot.exp_map's arithmetic after numpy's norm /
 * sin / cos (theta = |v|, S = sin(theta / 2), Cc = cos(theta / 2), all [t1 - t0, F]): k = theta > 1e-12 ? S / max(theta,
 * 1e-30) : 0.5; q = (v k, Cc) rounded to float32 */
void dl_v2_write_rot(const dl_v2 *v, i64 t0, i64 t1, const double *V, const double *theta, const double *S, const double *Cc, float *loc)
{
    const i64 F = v->F, B = v->B;
    for (i64 i = t0; i < t1; i++) {
        const i64 b = v->bone[i], r = (i - t0) * F;
        for (i64 f = 0; f < F; f++) {
            const double th = theta[r + f];
            const double k = th > 1e-12 ? S[r + f] / (th > 1e-30 ? th : 1e-30) : 0.5;
            const double *x = V + (r + f) * 3;
            float *o = loc + (f * B + b) * 7;
            o[0] = (float)(x[0] * k); o[1] = (float)(x[1] * k); o[2] = (float)(x[2] * k); o[3] = (float)Cc[r + f];
        }
    }
}

/* numpy's |v| over the last axis of 3 (np.linalg.norm: sqrt(add.reduce(v * v))): the two possible summation orders */
static inline double norm3(const double *x, int variant)
{
    const double a = x[0] * x[0], b = x[1] * x[1], c = x[2] * x[2];
    return sqrt(variant ? a + (b + c) : (a + b) + c);
}

void dl_norm3(const double *V, i64 n, int variant, double *out) { for (i64 i = 0; i < n; i++) out[i] = norm3(V + 3 * i, variant); }

/* the same with numpy's own float64 sin / cos inner loops (fast/nploops.py: self-tested pointers of this process)
 * on each track's half-angles: exactly np.sin / np.cos whatever numpy uses on this host */
typedef void (*np_loop)(char **args, const int64_t *dims, const int64_t *steps, void *data);
i64 dl_v2_write_rot_np(const dl_v2 *v, i64 t0, i64 t1, const double *V, int variant, void *sin_fn, void *sin_dt, void *cos_fn,
                       void *cos_dt, float *loc)
{
    const i64 F = v->F, B = v->B;
    double *th = (double *)malloc(sizeof(double) * (size_t)(4 * (F > 0 ? F : 1)));
    if (!th) return DL_E_ALLOC;
    double *hf = th + F, *S = th + 2 * F, *Cc = th + 3 * F;
    const int64_t dims[1] = {F}, steps[2] = {8, 8};
    for (i64 i = t0; i < t1; i++) {
        const i64 b = v->bone[i], r = (i - t0) * F;
        for (i64 f = 0; f < F; f++) { th[f] = norm3(V + (r + f) * 3, variant); hf[f] = 0.5 * th[f]; }
        char *as[2] = {(char *)hf, (char *)S}, *ac[2] = {(char *)hf, (char *)Cc};
        ((np_loop)sin_fn)(as, dims, steps, sin_dt);
        ((np_loop)cos_fn)(ac, dims, steps, cos_dt);
        for (i64 f = 0; f < F; f++) {
            const double t_ = th[f];
            const double k = t_ > 1e-12 ? S[f] / (t_ > 1e-30 ? t_ : 1e-30) : 0.5;
            const double *x = V + (r + f) * 3;
            float *o = loc + (f * B + b) * 7;
            o[0] = (float)(x[0] * k); o[1] = (float)(x[1] * k); o[2] = (float)(x[2] * k); o[3] = (float)Cc[f];
        }
    }
    free(th);
    return 0;
}

/* translation tracks t0 .. t1 - 1: loc[:, bone, 4:7] = float32(V) */
void dl_v2_write_trans(const dl_v2 *v, i64 t0, i64 t1, const double *V, float *loc)
{
    const i64 F = v->F, B = v->B;
    for (i64 i = t0; i < t1; i++) {
        const i64 b = v->bone[i], r = (i - t0) * F;
        for (i64 f = 0; f < F; f++) {
            const double *x = V + (r + f) * 3;
            float *o = loc + (f * B + b) * 7 + 4;
            o[0] = (float)x[0]; o[1] = (float)x[1]; o[2] = (float)x[2];
        }
    }
}
