"""The integer transformer step on a CUDA GPU (tfgpu.cu): the C step's signature and results (integer operations only).
Opt-in for the decoder with CURVECODEC_TF_GPU=<device>; built with nvcc (NVCC, else /usr/local/cuda/bin/nvcc) for
TFGPU_ARCH (default sm_89). Device = the CUDA ordinal inside CUDA_VISIBLE_DEVICES.
"""
import ctypes
import os
import subprocess
import tempfile
import weakref

import numpy as np

from ._build import HERE, _tag, build_dir

ARCH = os.environ.get('TFGPU_ARCH', 'sm_89')
NVCC = os.environ.get('NVCC', '/usr/local/cuda/bin/nvcc')
FLAGS = ['-O3', f'-arch={ARCH}', '-Xcompiler', '-fPIC', '-shared', '-std=c++17']
_vp = ctypes.c_void_p
_i64 = ctypes.c_int64
_lib = []


def lib():
    if not _lib:
        so = os.path.join(build_dir(), _tag('tfgpu', ['tfgpu.cu'], FLAGS))
        if not os.path.exists(so):
            fd, tmp = tempfile.mkstemp(suffix='.so', dir=os.path.dirname(so))
            os.close(fd)
            try:
                r = subprocess.run([NVCC, *FLAGS, '-o', tmp, os.path.join(HERE, 'tfgpu.cu')], capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError(r.stderr[-3000:])
                os.replace(tmp, so)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        L = ctypes.CDLL(so)
        L.tfg_model_create.restype = _vp; L.tfg_model_create.argtypes = [_vp, _vp, ctypes.c_int]
        L.tfg_model_free.restype = None; L.tfg_model_free.argtypes = [_vp]
        L.tfg_state_create.restype = _vp; L.tfg_state_create.argtypes = [_vp, _i64, _i64]
        L.tfg_state_free.restype = None; L.tfg_state_free.argtypes = [_vp]
        L.tfg_step.restype = ctypes.c_int; L.tfg_step.argtypes = [_vp, _i64, _i64, _vp, _vp, _vp]
        _lib.append(L)
    return _lib[0]


class _GpuModel:
    def __init__(self, g, device):
        from .tfstep import pack_model
        L = lib()
        self.L = L
        self._P, self._arrs, ptrs = pack_model(g)
        self.ptr = L.tfg_model_create(self._P.ctypes.data, ctypes.cast(ptrs, ctypes.c_void_p), int(device))
        if not self.ptr:
            raise RuntimeError('tfg_model_create failed')
        self._fin = weakref.finalize(self, L.tfg_model_free, self.ptr)


def c_model(g, C, pad, device=0):
    """(fn address, ctx address, release) of a fresh GPU step state (the contract of tfstep.c_model)."""
    m = g.__dict__.get('_gpu_model')
    if m is None:
        m = g._gpu_model = _GpuModel(g, device)
    ptr = m.L.tfg_state_create(m.ptr, int(C), int(pad))
    if not ptr:
        raise RuntimeError('tfg_state_create failed')
    fin = weakref.finalize(m, m.L.tfg_state_free, ptr)

    def release():
        fin()
    return ctypes.cast(m.L.tfg_step, ctypes.c_void_p).value, int(ptr), release


def selftest(g, C=8, steps=64, device=0):
    """The GPU step equals the numpy step on random features (returns True)."""
    rng = np.random.default_rng(0)
    fn, ctx, release = c_model(g, C, 128, device)
    step = ctypes.CFUNCTYPE(ctypes.c_int, _vp, _i64, _i64, _vp, _vp, _vp)(fn)
    g.start(C, 128)
    try:
        for t in range(steps):
            x = rng.integers(-3000, 3000, (C, 36)).astype(np.int64)
            act = np.arange(C, dtype=np.int64)
            out = np.empty((C, 3 * g.K), np.int64)
            if step(ctx, t, C, act.ctypes.data, x.ctypes.data, out.ctypes.data):
                return False
            ref = np.concatenate(g.step(x, act), axis=1)
            if not np.array_equal(out, ref):
                return False
        return True
    finally:
        release()
