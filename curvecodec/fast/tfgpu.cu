/* The integer transformer step of model.IntTf.step on a CUDA GPU, behind the same C signature as fast/tfstep.c:
 *
 *   int tfg_step(void *state, int64_t t, int64_t n_act, const int64_t *act, const int64_t *x, int64_t *out)
 *
 * One kernel launch per lockstep step: one thread block per active curve runs the whole step (embed -> L blocks -> head)
 * in shared memory; the KV caches of all C curves stay on the device (int32, [C][L][2][W][D]). Every operation is an
 * integer operation (int16 weights x int32 activations -> int64 sums, int64 shifts / clips / LayerNorm, the Q16 EXP_TAB,
 * int64 division for p), so the result does not depend on thread order or the GPU: bit-identical to IntTf.step (numpy's
 * float64 BLAS sums are exact integers, and so are these). The per-step hand-off: act / features / outputs travel through
 * mapped pinned host memory (the kernel reads and writes it directly), so a step is one memcpy of the inputs into the
 * pinned buffer, one launch, one stream synchronisation.
 *
 * Host API (ctypes; fast/tfgpu.py):
 *   void *tfg_model_create(const int64_t *P, const int64_t *const *A, int device)   same P / A layout as tfs_model_create
 *   void  tfg_model_free(void *model)
 *   void *tfg_state_create(void *model, int64_t C, int64_t pad)
 *   void  tfg_state_free(void *state)
 *   int   tfg_step(...)                                                              0 / -1 contract / -2 range / -3 CUDA
 */
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

typedef long long ll;
typedef unsigned long long ull;

#define MAXL 16
#define DMAX 128
#define FFMAX 512
#define HWMAX 1024          /* NH * W */
#define NFMAX 64
#define TPB 256
#define PARTN 2048         /* partial sums per matvec (S x Mp) */

struct GModel {
    int D, NL, NH, dh, W, K, NF, FF, FD, log2d;
    int ws0, wsf, wsh, wsq[MAXL], wso[MAXL], ws1[MAXL], ws2[MAXL];
    ll exp_lim, logs_lo, logs_hi, h_lim, ln_eps, rsq_bits, aq, hq;
    /* weights transposed to [N][M] (int16) so that threads j = output read consecutive words */
    const short *W0, *Wf, *Wh, *Wqkv[MAXL], *Wo[MAXL], *W1[MAXL], *W2[MAXL];
    const ll *b0, *bf, *bh, *bqkv[MAXL], *bo[MAXL], *b1[MAXL], *b2[MAXL];
    const int *rel;          /* [NL][NH][W] */
    const int *exp32;        /* [exp_lim + 1] */
    const ll *rsq;
};

struct HostModel {
    GModel g;
    int device;
    void *dmem[64]; int ndmem;
    ll tpl_pad; int *tpl;    /* device: one curve's cache after `pad` zero steps */
    ll curve_len;            /* ints per curve: NL * 2 * W * D */
};

struct HostState {
    HostModel *m;
    ll C, pad, t;
    int *kv;                 /* device [C][NL][2][W][D] */
    ll *h_act, *h_x, *h_out; /* mapped pinned */
    ll *d_act, *d_x, *d_out;
    int *h_err, *d_err;
    ll *last;
    cudaStream_t stream;
};

__device__ __forceinline__ ll clampi(ll v, ll lo, ll hi) { return v < lo ? lo : (v > hi ? hi : v); }

__device__ __forceinline__ ll inv_sqrt_q16_of_q24(const GModel &m, ll v)
{
    if (v < 1) v = 1;
    const ll e = 63 - __clzll(v);
    const ll e2 = e & ~1LL;
    const ll sh = e2 - m.rsq_bits;
    const ll mm = sh >= 0 ? (v >> sh) : (ll)((ull)v << (-sh));
    const ll r = m.rsq[mm - (1LL << m.rsq_bits)];
    const ll k = 12 - (e2 >> 1);
    return k >= 0 ? (ll)((ull)r << k) : (r >> (-k));
}

/* LayerNorm of h (D values) into a, by warp 0 (numpy's integer operations; the uint64 sum wraps like numpy's int64) */
__device__ void layer_norm(const GModel &m, const int *h, int *a, ll *sh_red)
{
    const int D = m.D, lane = threadIdx.x & 31;
    if (threadIdx.x < 32) {
        ll s = 0;
        for (int i = lane; i < D; i += 32) s += h[i];
        for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        s = __shfl_sync(0xffffffffu, s, 0);
        const ll mean = s >> m.log2d;
        ull v = 0;
        for (int i = lane; i < D; i += 32) { const ll c = (ll)h[i] - mean; v += (ull)(c * c); }
        for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        v = __shfl_sync(0xffffffffu, v, 0);
        const ll var = ((ll)v >> m.log2d) + m.ln_eps;
        const ll inv = inv_sqrt_q16_of_q24(m, var);
        for (int i = lane; i < D; i += 32) a[i] = (int)((((ll)h[i] - mean) * inv) >> 16);
    }
    (void)sh_red;
}

/* res[j] = sum_i W[i][j] in[i] for j < M with all threads of the block. W transposed [N][Mp] int16 (Mp = M rounded up to
 * 8): a thread owns 8 consecutive outputs (one int4 load = 8 weights per input) and a strided share of the inputs, so
 * every thread has its loads in flight at once; the S partial sums per output meet in part[] (exact int64: the order of
 * an exact integer sum does not matter). Ends with the block synchronised and res[] valid. */
__device__ __forceinline__ void matvec(const short *__restrict__ WT, int N, int M, const int *in, ll *part, ll *res)
{
    const int tid = threadIdx.x, nt = blockDim.x;
    const int Mp = (M + 7) & ~7, G = Mp >> 3;
    int S = nt / G; if (S < 1) S = 1; if (S > N) S = N; if (S * Mp > PARTN) S = PARTN / Mp;
    for (int k = tid; k < G * S; k += nt) {
        const int g = k % G, sidx = k / G;
        ll acc[8];
#pragma unroll
        for (int q = 0; q < 8; q++) acc[q] = 0;
        int i = sidx;
        for (; i + 7 * S < N; i += 8 * S) {
            int4 w4[8];
#pragma unroll
            for (int u = 0; u < 8; u++) w4[u] = __ldg((const int4 *)(WT + (size_t)(i + u * S) * Mp) + g);
#pragma unroll
            for (int u = 0; u < 8; u++) {
                const int x = in[i + u * S];
                const short *ws = (const short *)&w4[u];
#pragma unroll
                for (int q = 0; q < 8; q++) acc[q] += (ll)ws[q] * (ll)x;
            }
        }
        for (; i < N; i += S) {
            const int4 w4 = __ldg((const int4 *)(WT + (size_t)i * Mp) + g);
            const int x = in[i];
            const short *ws = (const short *)&w4;
#pragma unroll
            for (int q = 0; q < 8; q++) acc[q] += (ll)ws[q] * (ll)x;
        }
#pragma unroll
        for (int q = 0; q < 8; q++) part[(size_t)sidx * Mp + 8 * g + q] = acc[q];
    }
    __syncthreads();
    for (int j = tid; j < M; j += nt) {
        ll acc = 0;
        for (int sidx = 0; sidx < S; sidx++) acc += part[(size_t)sidx * Mp + j];
        res[j] = acc;
    }
    __syncthreads();
}

__global__ void __launch_bounds__(TPB) tf_step_kernel(GModel m, ll T, int n, const ll *act, const ll *x, ll *out, int *kv,
                                                      ll curve_len, int *err)
{
    __shared__ int h[DMAX], a[FFMAX], qkv[3 * DMAX], o[DMAX], u[FFMAX];
    __shared__ ll lg[HWMAX];
    __shared__ ll part[PARTN], res[FFMAX];
    __shared__ int xin[NFMAX];
    __shared__ ll mx[32], tot[32];
    const int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int D = m.D, W = m.W, NH = m.NH, dh = m.dh, FF = m.FF;
    const ll c = act[b];
    int *cache = kv + (size_t)c * curve_len;
    const int slot = (int)(T % W);                  /* ring offset of position w: off = slot - w (mod W); valid = off <= T */
    const bool full = T >= W - 1;
    for (int i = tid; i < m.NF; i += nt) {
        const ll v = x[(size_t)b * m.NF + i];
        if (v < -(1LL << 30) || v > (1LL << 30)) atomicExch(err, 1);
        xin[i] = (int)v;
    }
    __syncthreads();
    matvec(m.W0, m.NF, D, xin, part, res);
    for (int j = tid; j < D; j += nt) h[j] = (int)(clampi((res[j] + m.b0[j]) >> (m.ws0 - m.aq), 0, 1LL << 24) << (m.hq - m.aq));
    __syncthreads();
    for (int l = 0; l < m.NL; l++) {
        int *Kc = cache + (size_t)l * 2 * W * D, *Vc = Kc + (size_t)W * D;
        layer_norm(m, h, a, nullptr);
        __syncthreads();
        const int shq = m.wsq[l] + (int)(m.hq - m.aq);
        matvec(m.Wqkv[l], D, 3 * D, a, part, res);
        for (int j = tid; j < 3 * D; j += nt) {
            const ll v = (res[j] + m.bqkv[l][j]) >> shq;
            if (v < INT32_MIN || v > INT32_MAX) atomicExch(err, 2);
            qkv[j] = (int)v;
            if (j >= D && j < 2 * D) Kc[(size_t)slot * D + (j - D)] = (int)v;
            else if (j >= 2 * D) Vc[(size_t)slot * D + (j - 2 * D)] = (int)v;
        }
        __syncthreads();
        /* logits */
        const int *rel = m.rel + (size_t)l * NH * W;
        for (int idx = tid; idx < NH * W; idx += nt) {
            const int hh = idx / W, w = idx - hh * W;
            int off = slot - w; if (off < 0) off += W;
            ll v;
            if (full || off <= T) {
                ll d = 0;
                const int *kr = Kc + (size_t)w * D + hh * dh;
                const int *q = qkv + hh * dh;
                if (dh == 16) {
                    const int4 *k4 = (const int4 *)kr;
#pragma unroll
                    for (int i4 = 0; i4 < 4; i4++) {
                        const int4 kv4 = k4[i4];
                        d += (ll)q[4 * i4] * kv4.x + (ll)q[4 * i4 + 1] * kv4.y + (ll)q[4 * i4 + 2] * kv4.z + (ll)q[4 * i4 + 3] * kv4.w;
                    }
                } else {
                    for (int i = 0; i < dh; i++) d += (ll)q[i] * (ll)kr[i];
                }
                v = (d >> 8) + rel[hh * W + off];
            } else v = -(1LL << 40);
            lg[idx] = v;
        }
        __syncthreads();
        /* softmax per head: warp hh */
        const int warp = tid >> 5, lane = tid & 31;
        for (int hh = warp; hh < NH; hh += nt >> 5) {
            ll mv = LLONG_MIN;
            for (int w = lane; w < W; w += 32) mv = max(mv, lg[hh * W + w]);
            for (int oo = 16; oo; oo >>= 1) mv = max(mv, __shfl_down_sync(0xffffffffu, mv, oo));
            mv = __shfl_sync(0xffffffffu, mv, 0);
            ll ts = 0;
            for (int w = lane; w < W; w += 32) {
                int off = slot - w; if (off < 0) off += W;
                const ll e = (full || off <= T) ? (ll)__ldg(m.exp32 + (clampi(lg[hh * W + w] - mv, -m.exp_lim, 0) + m.exp_lim)) : 0;
                lg[hh * W + w] = e; ts += e;
            }
            for (int oo = 16; oo; oo >>= 1) ts += __shfl_down_sync(0xffffffffu, ts, oo);
            if (lane == 0) { mx[hh] = mv; tot[hh] = ts; }
        }
        __syncthreads();
        /* p = (e << 16) // tot: float estimate within 1 of the floor (two float roundings, quotient <= 2^16), corrected by the
         * exact remainder computed modulo 2^32 (|r| < 2 tot < 2^24) -- the CPU step's argument (tfstep.c sm_p) */
        for (int idx = tid; idx < NH * W; idx += nt) {
            const int e = (int)lg[idx], tt = (int)tot[idx / W];
            int q = (int)((float)e * (65536.0f / (float)tt));
            const int r = (int)(((unsigned)e << 16) - (unsigned)q * (unsigned)tt);
            if (r < 0) q--; else if (r >= tt) q++;
            lg[idx] = q;
        }
        __syncthreads();
        /* o = (sum_w p v) >> 16 */
        {
            /* a thread owns 4 consecutive dims (one int4 of the V row, same head since dh % 4 == 0) and every S-th position */
            const int G = D >> 2;
            int S = nt / G; if (S < 1) S = 1; if (S > W) S = W; if (S * D > PARTN) S = PARTN / D;
            for (int k = tid; k < G * S; k += nt) {
                const int g = k % G, sidx = k / G, hh = (4 * g) / dh;
                ll s4[4] = {0, 0, 0, 0};
                int w = sidx;
                for (; w + 7 * S < W; w += 8 * S) {
                    int4 v4[8];
#pragma unroll
                    for (int uu = 0; uu < 8; uu++) v4[uu] = *((const int4 *)(Vc + (size_t)(w + uu * S) * D) + g);
#pragma unroll
                    for (int uu = 0; uu < 8; uu++) {
                        const ll pw = (ll)(int)lg[hh * W + w + uu * S];
                        s4[0] += pw * (ll)v4[uu].x; s4[1] += pw * (ll)v4[uu].y; s4[2] += pw * (ll)v4[uu].z; s4[3] += pw * (ll)v4[uu].w;
                    }
                }
                for (; w < W; w += S) {
                    const int4 v4 = *((const int4 *)(Vc + (size_t)w * D) + g);
                    const ll pw = (ll)(int)lg[hh * W + w];
                    s4[0] += pw * (ll)v4.x; s4[1] += pw * (ll)v4.y; s4[2] += pw * (ll)v4.z; s4[3] += pw * (ll)v4.w;
                }
#pragma unroll
                for (int q = 0; q < 4; q++) part[(size_t)sidx * D + 4 * g + q] = s4[q];
            }
            __syncthreads();
            for (int j = tid; j < D; j += nt) {
                ll s = 0;
                for (int sidx = 0; sidx < S; sidx++) s += part[(size_t)sidx * D + j];
                o[j] = (int)(s >> 16);
            }
        }
        __syncthreads();
        const int sho = m.wso[l] - (int)(m.hq - m.aq);
        matvec(m.Wo[l], D, D, o, part, res);
        for (int j = tid; j < D; j += nt) h[j] = (int)clampi((ll)h[j] + ((res[j] + m.bo[l][j]) >> sho), -m.h_lim, m.h_lim);
        __syncthreads();
        layer_norm(m, h, a, nullptr);
        __syncthreads();
        const int sh1 = m.ws1[l] + (int)(m.hq - m.aq);
        matvec(m.W1[l], D, FF, a, part, res);
        for (int j = tid; j < FF; j += nt) u[j] = (int)clampi((res[j] + m.b1[l][j]) >> sh1, 0, 1LL << 24);
        __syncthreads();
        const int sh2 = m.ws2[l] - (int)(m.hq - m.aq);
        matvec(m.W2[l], FF, D, u, part, res);
        for (int j = tid; j < D; j += nt) h[j] = (int)clampi((ll)h[j] + ((res[j] + m.b2[l][j]) >> sh2), -m.h_lim, m.h_lim);
        __syncthreads();
    }
    const int shf = m.wsf + (int)(m.hq - m.aq);
    matvec(m.Wf, D, m.FD, h, part, res);
    for (int j = tid; j < m.FD; j += nt) u[j] = (int)clampi((res[j] + m.bf[j]) >> shf, 0, 1LL << 24);
    __syncthreads();
    matvec(m.Wh, m.FD, 3 * m.K, u, part, res);
    for (int j = tid; j < 3 * m.K; j += nt) {
        const ll v = (res[j] + m.bh[j]) >> m.wsh;
        out[(size_t)b * 3 * m.K + j] = j >= 2 * m.K ? clampi(v, m.logs_lo, m.logs_hi) : v;
    }
}

__global__ void replicate_kernel(int *kv, const int *tpl, ll C, ll len)
{
    const ll total = C * len;
    for (ll i = (ll)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (ll)gridDim.x * blockDim.x) kv[i] = tpl[i % len];
}

/* ------------------------------------------------------------------------------------------------ host */
extern "C" {

static void *dupload(HostModel *hm, const void *src, size_t bytes)
{
    void *d = nullptr;
    if (cudaMalloc(&d, bytes ? bytes : 8) != cudaSuccess) return nullptr;
    if (bytes && cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(d); return nullptr; }
    hm->dmem[hm->ndmem++] = d;
    return d;
}

static const short *up_wT(HostModel *hm, const int64_t *W, int M, int N, int *bad)
{
    const int Mp = (M + 7) & ~7;                      /* rows padded to whole int4 (8 shorts) */
    short *t = (short *)calloc((size_t)Mp * N, sizeof(short));
    for (int j = 0; j < M; j++)
        for (int i = 0; i < N; i++) {
            const int64_t w = W[(size_t)j * N + i];
            if (w < -32768 || w > 32767) *bad = 1;
            t[(size_t)i * Mp + j] = (short)w;
        }
    const short *d = (const short *)dupload(hm, t, (size_t)Mp * N * sizeof(short));
    free(t);
    if (!d) *bad = 1;
    return d;
}

static const ll *up_b(HostModel *hm, const int64_t *b, int M, int *bad)
{
    const ll *d = (const ll *)dupload(hm, b, (size_t)M * sizeof(ll));
    if (!d) *bad = 1;
    return d;
}

void tfg_model_free(void *vm);

void *tfg_model_create(const int64_t *P, const int64_t *const *A, int device)
{
    if (cudaSetDevice(device) != cudaSuccess) return nullptr;
    cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost);   /* spin in the per-step sync (lower wake-up latency); fails harmlessly if the context exists */
    HostModel *hm = (HostModel *)calloc(1, sizeof(HostModel));
    GModel &m = hm->g;
    hm->device = device;
    m.D = (int)P[0]; m.NL = (int)P[1]; m.NH = (int)P[2]; m.W = (int)P[3]; m.K = (int)P[4]; m.NF = (int)P[5];
    m.FF = (int)P[6]; m.FD = (int)P[7]; m.ws0 = (int)P[8]; m.wsf = (int)P[9]; m.wsh = (int)P[10];
    m.exp_lim = P[11]; m.logs_lo = P[12]; m.logs_hi = P[13]; m.h_lim = P[14]; m.ln_eps = P[15]; m.rsq_bits = P[16];
    const ll n_rsq = P[17]; m.aq = P[18]; m.hq = P[19];
    if (m.NL < 1 || m.NL > MAXL || m.D > DMAX || m.FF > FFMAX || m.FD > FFMAX || m.NH * m.W > HWMAX || m.NF > NFMAX ||
        m.NH > 32 || m.D % m.NH || 3 * m.K > FFMAX || (m.D / m.NH) % 4) { free(hm); return nullptr; }
    m.dh = m.D / m.NH;
    m.log2d = 0; while ((1 << m.log2d) < m.D) m.log2d++;
    for (int l = 0; l < m.NL; l++) { m.wsq[l] = (int)P[20 + 4 * l]; m.wso[l] = (int)P[21 + 4 * l]; m.ws1[l] = (int)P[22 + 4 * l]; m.ws2[l] = (int)P[23 + 4 * l]; }
    int bad = 0;
    m.W0 = up_wT(hm, A[0], m.D, m.NF, &bad); m.b0 = up_b(hm, A[1], m.D, &bad);
    m.Wf = up_wT(hm, A[2], m.FD, m.D, &bad); m.bf = up_b(hm, A[3], m.FD, &bad);
    m.Wh = up_wT(hm, A[4], 3 * m.K, m.FD, &bad); m.bh = up_b(hm, A[5], 3 * m.K, &bad);
    int *e32 = (int *)malloc((size_t)(m.exp_lim + 1) * sizeof(int));
    for (ll k = 0; k <= m.exp_lim; k++) e32[k] = (int)A[6][k];
    m.exp32 = (const int *)dupload(hm, e32, (size_t)(m.exp_lim + 1) * sizeof(int)); free(e32);
    m.rsq = (const ll *)dupload(hm, A[7], (size_t)n_rsq * sizeof(ll));
    int *rel = (int *)malloc((size_t)m.NL * m.NH * m.W * sizeof(int));
    for (int l = 0; l < m.NL; l++) {
        const int64_t *const *L = A + 8 + 9 * l;
        m.Wqkv[l] = up_wT(hm, L[0], 3 * m.D, m.D, &bad); m.bqkv[l] = up_b(hm, L[1], 3 * m.D, &bad);
        m.Wo[l] = up_wT(hm, L[2], m.D, m.D, &bad); m.bo[l] = up_b(hm, L[3], m.D, &bad);
        m.W1[l] = up_wT(hm, L[4], m.FF, m.D, &bad); m.b1[l] = up_b(hm, L[5], m.FF, &bad);
        m.W2[l] = up_wT(hm, L[6], m.D, m.FF, &bad); m.b2[l] = up_b(hm, L[7], m.D, &bad);
        for (int k = 0; k < m.NH * m.W; k++) rel[(size_t)l * m.NH * m.W + k] = (int)L[8][k];
    }
    m.rel = (const int *)dupload(hm, rel, (size_t)m.NL * m.NH * m.W * sizeof(int)); free(rel);
    if (bad || !m.exp32 || !m.rsq || !m.rel) { tfg_model_free(hm); return nullptr; }
    hm->curve_len = (ll)m.NL * 2 * m.W * m.D;
    hm->tpl_pad = -1;
    return hm;
}

void tfg_model_free(void *vm)
{
    HostModel *hm = (HostModel *)vm;
    if (!hm) return;
    cudaSetDevice(hm->device);
    for (int i = 0; i < hm->ndmem; i++) cudaFree(hm->dmem[i]);
    if (hm->tpl) cudaFree(hm->tpl);
    free(hm);
}

void tfg_state_free(void *vs)
{
    HostState *s = (HostState *)vs;
    if (!s) return;
    cudaSetDevice(s->m->device);
    if (s->stream) cudaStreamSynchronize(s->stream);
    if (s->kv) cudaFree(s->kv);
    if (s->h_act) cudaFreeHost(s->h_act);
    if (s->h_x) cudaFreeHost(s->h_x);
    if (s->h_out) cudaFreeHost(s->h_out);
    if (s->h_err) cudaFreeHost(s->h_err);
    if (s->stream) cudaStreamDestroy(s->stream);
    free(s->last);
    free(s);
}

static int launch_step(HostState *s, ll T, ll n)
{
    const GModel &m = s->m->g;
    *s->h_err = 0;
    tf_step_kernel<<<(unsigned)n, TPB, 0, s->stream>>>(m, T, (int)n, s->d_act, s->d_x, s->d_out, s->kv, s->m->curve_len, s->d_err);
    if (cudaGetLastError() != cudaSuccess) return -3;
    if (cudaStreamSynchronize(s->stream) != cudaSuccess) return -3;
    return *(volatile int *)s->h_err ? -2 : 0;
}

static HostState *state_alloc(HostModel *hm, ll C)
{
    HostState *s = (HostState *)calloc(1, sizeof(HostState));
    s->m = hm; s->C = C;
    const GModel &m = hm->g;
    const ll Cc = C > 0 ? C : 1;
    bool ok = cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking) == cudaSuccess;
    ok = ok && cudaMalloc(&s->kv, (size_t)Cc * hm->curve_len * sizeof(int)) == cudaSuccess;
    ok = ok && cudaHostAlloc((void **)&s->h_act, (size_t)Cc * sizeof(ll), cudaHostAllocMapped) == cudaSuccess;
    ok = ok && cudaHostAlloc((void **)&s->h_x, (size_t)Cc * m.NF * sizeof(ll), cudaHostAllocMapped) == cudaSuccess;
    ok = ok && cudaHostAlloc((void **)&s->h_out, (size_t)Cc * 3 * m.K * sizeof(ll), cudaHostAllocMapped) == cudaSuccess;
    ok = ok && cudaHostAlloc((void **)&s->h_err, sizeof(int), cudaHostAllocMapped) == cudaSuccess;
    if (ok) {
        cudaHostGetDevicePointer((void **)&s->d_act, s->h_act, 0);
        cudaHostGetDevicePointer((void **)&s->d_x, s->h_x, 0);
        cudaHostGetDevicePointer((void **)&s->d_out, s->h_out, 0);
        cudaHostGetDevicePointer((void **)&s->d_err, s->h_err, 0);
        ok = cudaMemset(s->kv, 0, (size_t)Cc * hm->curve_len * sizeof(int)) == cudaSuccess;
    }
    s->last = (ll *)malloc((size_t)Cc * sizeof(ll));
    for (ll c = 0; c < Cc; c++) s->last[c] = -1;
    if (!ok) { tfg_state_free(s); return nullptr; }
    return s;
}

void *tfg_state_create(void *vm, int64_t C, int64_t pad)
{
    HostModel *hm = (HostModel *)vm;
    if (!hm || C < 0 || pad < 0) return nullptr;
    if (cudaSetDevice(hm->device) != cudaSuccess) return nullptr;
    const GModel &m = hm->g;
    if (pad > 0 && hm->tpl_pad != pad) {
        HostState *t1 = state_alloc(hm, 1);
        if (!t1) return nullptr;
        t1->h_act[0] = 0;
        for (int i = 0; i < m.NF; i++) t1->h_x[i] = 0;
        for (ll T = 0; T < pad; T++)
            if (launch_step(t1, T, 1)) { tfg_state_free(t1); return nullptr; }
        if (!hm->tpl && cudaMalloc(&hm->tpl, (size_t)hm->curve_len * sizeof(int)) != cudaSuccess) { tfg_state_free(t1); return nullptr; }
        cudaMemcpy(hm->tpl, t1->kv, (size_t)hm->curve_len * sizeof(int), cudaMemcpyDeviceToDevice);
        tfg_state_free(t1);
        hm->tpl_pad = pad;
    }
    HostState *s = state_alloc(hm, C);
    if (!s) return nullptr;
    s->pad = pad; s->t = 0;
    if (pad > 0 && C > 0) {
        replicate_kernel<<<256, 256, 0, s->stream>>>(s->kv, hm->tpl, C, hm->curve_len);
        if (cudaStreamSynchronize(s->stream) != cudaSuccess) { tfg_state_free(s); return nullptr; }
    }
    return s;
}

int tfg_step(void *vs, int64_t t, int64_t n_act, const int64_t *act, const int64_t *x, int64_t *out)
{
    HostState *s = (HostState *)vs;
    if (!s || t != s->t || n_act < 0 || n_act > s->C) return -1;
    for (ll i = 0; i < n_act; i++) {
        const ll c = act[i];
        if (c < 0 || c >= s->C || (i && c <= act[i - 1]) || s->last[c] != t - 1) return -1;
    }
    const GModel &m = s->m->g;
    if (n_act > 0) {
        memcpy(s->h_act, act, (size_t)n_act * sizeof(ll));
        memcpy(s->h_x, x, (size_t)n_act * m.NF * sizeof(ll));
        const int rc = launch_step(s, s->pad + t, n_act);
        if (rc) return rc;
        memcpy(out, s->h_out, (size_t)n_act * 3 * m.K * sizeof(ll));
    }
    for (ll i = 0; i < n_act; i++) s->last[act[i]] = t;
    s->t = t + 1;
    return 0;
}

}  /* extern "C" */
