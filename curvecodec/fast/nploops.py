"""numpy's own float64 ufunc inner loops (sin, cos, log2) as C function pointers, self-tested.

numpy's SIMD sin / cos / log2 are not libm's, so a C kernel that must reproduce numpy's results bit for bit calls
numpy's inner loop directly:

    typedef void (*ufunc_loop)(char **args, const int64_t *dims, const int64_t *steps, void *data);
    char *args[2] = {(char *)in, (char *)out}; int64_t dims[1] = {n}, steps[2] = {8, 8};
    loop(args, dims, steps, data);               /* out[i] = np.sin(in[i]), contiguous float64 */

`loops()` returns None unless the self-test passes: the pointer is numpy's first float64 -> float64 loop of the ufunc
(the one numpy's type resolution picks for float64 input), called on 72 lengths x 4 alignments and compared with the
ufunc itself. The pointers are valid for the lifetime of the process.
"""
import ctypes

import numpy as np

_LOOP = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_ssize_t), ctypes.POINTER(ctypes.c_ssize_t),
                         ctypes.c_void_p)
_CACHE = {}


class _UFunc(ctypes.Structure):
    """The head of numpy's PyUFuncObject (numpy/ufuncobject.h)."""
    _fields_ = [('ob_refcnt', ctypes.c_ssize_t), ('ob_type', ctypes.c_void_p), ('nin', ctypes.c_int), ('nout', ctypes.c_int),
                ('nargs', ctypes.c_int), ('identity', ctypes.c_int), ('functions', ctypes.c_void_p), ('data', ctypes.c_void_p),
                ('ntypes', ctypes.c_int), ('reserved1', ctypes.c_int), ('name', ctypes.c_char_p), ('types', ctypes.c_void_p)]


def _dd_loop(uf):
    """(loop pointer, data pointer or 0) of the first float64 -> float64 inner loop of a unary ufunc."""
    u = _UFunc.from_address(id(uf))
    if (u.nin, u.nout, u.nargs) != (1, 1, 2) or u.name != uf.__name__.encode() or not u.functions or not u.types:
        raise RuntimeError(f'unexpected ufunc layout for {uf.__name__}')
    types = ctypes.string_at(u.types, u.ntypes * 2)
    D = np.dtype(np.float64).num
    ks = [k for k in range(u.ntypes) if types[2 * k] == D and types[2 * k + 1] == D]
    if not ks:
        raise RuntimeError(f'no float64 loop in {uf.__name__}')
    fn = ctypes.c_void_p.from_address(u.functions + ctypes.sizeof(ctypes.c_void_p) * ks[0]).value
    dt = ctypes.c_void_p.from_address(u.data + ctypes.sizeof(ctypes.c_void_p) * ks[0]).value if u.data else None
    if not fn:
        raise RuntimeError(f'null loop in {uf.__name__}')
    return int(fn), int(dt or 0)


def _call(fn, dt, x):
    x = np.ascontiguousarray(x, np.float64)
    out = np.empty_like(x)
    args = (ctypes.c_void_p * 2)(x.ctypes.data, out.ctypes.data)
    dims = (ctypes.c_ssize_t * 1)(len(x)); steps = (ctypes.c_ssize_t * 2)(8, 8)
    _LOOP(fn)(args, dims, steps, dt or None)
    return out


def _selftest(name, fn, dt):
    rng = np.random.default_rng(20260928)
    special = [0.0, 1e-300, 5e-7, 1e-12, np.pi / 2, np.pi, 7.0, 40.0]
    uf = getattr(np, name)
    for n in list(range(1, 67)) + [100, 255, 256, 257, 1000, 4097]:
        if name == 'log2':
            x = np.concatenate([1.0 + 2.0 * np.abs(np.round(rng.normal(0, 30, n))), [1.0, 3.0, 2.5, 1e-300, 1e300]])
        else:
            x = np.concatenate([rng.random(n) * 3.3, rng.normal(0, 10, n), special])
        for off in range(4):
            xs = np.ascontiguousarray(np.concatenate([np.zeros(off), x]))[off:]
            if not np.array_equal(_call(fn, dt, xs), uf(xs)):
                return False
    return True


def loops(names=('sin', 'cos')):
    """{name: (loop pointer, data pointer)} of numpy's float64 loops, or None when any of them fails its self-test."""
    out = {}
    for name in names:
        if name not in _CACHE:
            try:
                fn, dt = _dd_loop(getattr(np, name))
                _CACHE[name] = (fn, dt) if _selftest(name, fn, dt) else None
            except Exception:  # noqa: BLE001 -- an unusual numpy build: no pointers
                _CACHE[name] = None
        if _CACHE[name] is None:
            return None
        out[name] = _CACHE[name]
    return out
