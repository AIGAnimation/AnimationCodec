"""Bit-exact C paths of the codec. The numpy code in the package is the reference; every function here returns the
identical result, or None when its C path does not apply (C disabled, build failed, an unusual array layout), and the
caller then runs its numpy code. The C sources are compiled on first use with the system C compiler
(cc -O3 -march=native -ffp-contract=off -fno-fast-math) and loaded with ctypes:

  kernels.c  (fastkern.c + enckern.c)  the encoder's search loops, object-space errors, NLMS, the adaptive model,
                                       interpolation, frequency tables, the residual features
  decloop.c                            the whole decoder loop (header, lockstep residuals, reconstruction)
  tfstep.c                             the integer transformer step (decoder) and the encoder's model pass
  tfgpu.cu                             the transformer step on a CUDA GPU (opt-in)

Switches (environment, read at import):
  CURVECODEC_NO_C=1                   numpy everywhere (nothing is built or loaded)
  CURVECODEC_FAST_OFF=a,b,..          parts off: search (encoder search loops), decoder, tfstep
  CURVECODEC_TF_THREADS=n             the transformer step's active curves shared by n threads (identical results)
  CURVECODEC_TF_GPU=k                 the decoder's transformer step on CUDA device k (needs nvcc; TFGPU_ARCH, default sm_89)
  CURVECODEC_BUILD_DIR=path           where the compiled libraries go (default: fast/_build, else ~/.cache/curvecodec)
`python -m curvecodec.fast` prints which paths are live.
"""
import os

ENABLED = os.environ.get('CURVECODEC_NO_C') != '1'
OFF = set(filter(None, os.environ.get('CURVECODEC_FAST_OFF', '').split(',')))


def _kern():
    if not ENABLED:
        return None
    from . import kernels
    return kernels if kernels.ready() else None


def _search():
    k = _kern()
    return k if k is not None and 'search' not in OFF and k.SEARCH else None


# ---------------------------------------------------------------- stateless kernels
def model_forward(sym, ctx, nsym, nctx, adapt_rate, init):
    k = _kern()
    return None if k is None else k.model_forward(sym, ctx, nsym, nctx, adapt_rate, init)


def nlms_run(A, x0, order, mu_shift, decode, init):
    k = _kern()
    return None if k is None else k.nlms_run(A, x0, order, mu_shift, decode, init)


def symbol_freqs(logw, mu, logs, alphabet):
    k = _kern()
    return None if k is None else k.symbol_freqs(logw, mu, logs, alphabet)


def interp_vectors(K, vk, F):
    k = _kern()
    return None if k is None else k.interp_vectors(K, vk, F)


def reconstruct(K, nk, step, F, kind):
    k = _search()
    return None if k is None else k.reconstruct(K, nk, step, F, kind)


# ---------------------------------------------------------------- the encoder's search
def eval_subtree_mean(S, b, new_local, early_exit, mlim, glim):
    k = _kern()
    return None if k is None else k.eval_subtree_mean(S, b, new_local, early_exit, mlim, glim)


def eval_subtree_frames(S, b, local_b, idx):
    k = _kern()
    return None if k is None else k.eval_subtree_frames(S, b, local_b, idx)


def commit_rows(S):
    k = _search()
    return False if k is None else k.commit_rows(S)


def remove_keys(S, tr, st, step, v, lam, reach, stats):
    k = _search()
    return False if k is None else k.remove_keys(S, tr, st, step, v, lam, reach, stats)


def grow_step(S, tr, st, v, grid, grow, lam, sizes, gbits, max_offers):
    k = _search()
    return False if k is None else k.grow_step(S, tr, st, v, grid, grow, lam, sizes, gbits, max_offers)


def ladder_c(S, kf_tracks, steps, vals, keys, nks, m_p, stats, lam0, ratio, max_levels, reach, gap_bits, flag_bits):
    k = _search()
    return None if k is None else k.ladder_c(S, kf_tracks, steps, vals, keys, nks, m_p, stats, lam0, ratio, max_levels, reach,
                                             gap_bits, flag_bits)


# ---------------------------------------------------------------- the transformer and the decoder
def tf_state(model, C, pad):
    if not ENABLED or 'tfstep' in OFF:
        return None
    from . import tfstep
    return tfstep.state(model, C, pad)


def tf_pack(model, st, n, n_max, G, R, Xl, pad):
    if not ENABLED or 'tfstep' in OFF or _kern() is None:
        return None
    from . import tfstep
    return tfstep.pack_forward(model, st, n, n_max, G, R, Xl, pad)


def decode(codec, blob, parents, offsets, fps):
    if not ENABLED or 'decoder' in OFF or _kern() is None:
        return None
    from . import decoder
    return decoder.decode(codec, blob, parents, offsets, fps)


def status():
    """What is live on this machine."""
    out = dict(enabled=ENABLED, off=sorted(OFF))
    if ENABLED:
        from . import decoder, kernels, tfstep
        out.update(kernels=kernels.status(), decoder=decoder.status(), tfstep=tfstep.status())
    return out
