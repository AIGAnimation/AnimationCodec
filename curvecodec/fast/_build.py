"""Compile a C library once per machine / source version and load it with ctypes."""
import ctypes
import hashlib
import os
import platform
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = '-mcpu=native' if platform.machine().lower() in ('arm64', 'aarch64') else '-march=native'
# every float operation in the kernels is element-wise IEEE arithmetic (correctly rounded at any vector width), FMA
# contraction off and no reassociation, so vectorising for the host CPU cannot change a bit
CFLAGS = ['-O3', NATIVE, '-ffp-contract=off', '-fno-fast-math', '-fno-strict-aliasing', '-fPIC']


def build_dir():
    d = os.environ.get('CURVECODEC_BUILD_DIR') or os.path.join(HERE, '_build')
    try:
        os.makedirs(d, exist_ok=True)
        if os.access(d, os.W_OK):
            return d
    except OSError:
        pass
    d = os.path.join(os.path.expanduser('~'), '.cache', 'curvecodec')
    os.makedirs(d, exist_ok=True)
    return d


def _tag(name, srcs, flags):
    h = hashlib.sha256()
    for s in srcs:
        path, extra = (s, []) if isinstance(s, str) else s
        with open(os.path.join(HERE, path), 'rb') as f:
            h.update(f.read() + ' '.join(extra).encode())
    if any((s if isinstance(s, str) else s[0]) == 'decloop.c' for s in srcs):
        with open(os.path.join(HERE, 'decloop.h'), 'rb') as f:
            h.update(f.read())
    h.update(' '.join(flags).encode())
    h.update(platform.node().encode())          # -march=native: one build per machine (a shared file system)
    return f'{name}-{platform.system().lower()}-{platform.machine().lower()}-py{sys.version_info[0]}{sys.version_info[1]}-{h.hexdigest()[:12]}.so'


def build_lib(name, srcs, flags=(), std='c99', link=('-lm',)):
    """srcs: file names in this directory, or (file name, extra flags) compiled to their own object. -> ctypes.CDLL."""
    flags = list(CFLAGS) + [f'-std={std}'] + list(flags)
    so = os.path.join(build_dir(), _tag(name, srcs, flags))
    if not os.path.exists(so):
        cc = os.environ.get('CC', 'cc')
        fd, tmp = tempfile.mkstemp(suffix='.so', dir=os.path.dirname(so))
        os.close(fd)
        objs = []
        try:
            for k, s in enumerate(srcs):
                path, extra = (s, []) if isinstance(s, str) else s
                o = f'{tmp}.{k}.o'
                objs.append(o)
                subprocess.run([cc, *flags, *extra, '-c', '-o', o, os.path.join(HERE, path)], check=True, capture_output=True, text=True)
            subprocess.run([cc, *flags, '-shared', '-o', tmp, *objs, *link], check=True, capture_output=True, text=True)
            os.replace(tmp, so)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'building {name} failed: {e.stderr[-2000:]}') from None
        finally:
            for o in objs + [tmp]:
                if os.path.exists(o):
                    os.unlink(o)
    return ctypes.CDLL(so)
