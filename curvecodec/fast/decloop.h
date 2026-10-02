/* The model hook of the C decoder (decloop.c): the lockstep residual loop calls the probability model once per step
 *
 *     rc = step(ctx, t, n_act, act, x, out)
 *
 *   ctx    the model's own state (weights + per-curve caches for the C curves of the clip; tfstep.c tfs_state_create)
 *   t      lockstep step 0 .. n_max - 1
 *   n_act  active curves this step (curves with n_c > t); shrinks monotonically, never grows
 *   act    [n_act] curve ids in 0 .. C - 1, ascending
 *   x      [n_act, 36] int64 Q8 features (features.CurveState.features()), row i = curve act[i]
 *   out    [n_act, 3 K] int64, row i = logw[K] | mu[K] | logs[K] (logs clipped to LOGS_LO .. LOGS_HI): exactly the rows
 *          act of the (logw, mu, logs) model.IntTf.step returns
 *   rc     0 = ok; -1 = contract violation (t differs from the ctx's own counter, act not a subset of the previous step's
 *          act, an id >= C); -2 = a value outside the fast path's proven range; any other nonzero = error.
 *          On ANY nonzero return the ctx state is undefined: the caller decodes the whole clip again in numpy.
 *
 * A curve is active exactly for t < n_c, and every model row depends only on the curve's own past (the per-curve KV
 * cache), so an implementation may skip the inactive curves entirely.
 */
#ifndef DECLOOP_H
#define DECLOOP_H
#include <stdint.h>

typedef int (*dl_step_fn)(void *ctx, int64_t t, int64_t n_act, const int64_t *act, const int64_t *x, int64_t *out);

#define DL_NFEAT 36

#endif
