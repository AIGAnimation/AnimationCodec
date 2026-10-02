"""Bit I/O and the rANS entropy coder.

One blob carries a single rANS stream with two kinds of symbols, interleaved in decode order:
  * adaptive symbols: an 83-symbol alphabet with one adaptive frequency model per context (counts start from a prior
    table, +MODEL_INC per coded symbol, halved when a context's total exceeds its limit);
  * residual symbols with explicit (start, freq) pairs from the learned model (probability scale 2^15).
Both are coded with the same 32-bit rANS state (renormalisation by bytes, RANS_L = 2^23).
"""
import numpy as np

from . import fast as _fast

RANS_SCALE_BITS = 15
RANS_M = 1 << RANS_SCALE_BITS
RANS_L = 1 << 23
MODEL_INC = 32
ADAPT_RATE = 5


class BitWriter:
    """MSB-first bit packer; write(value, nbits) appends the low nbits of value."""

    __slots__ = ('_buf', '_acc', '_nacc', '_nbits')

    def __init__(self):
        self._buf = bytearray()
        self._acc = 0
        self._nacc = 0
        self._nbits = 0

    def write(self, value, nbits):
        if nbits <= 0:
            if nbits == 0:
                return
            raise ValueError('nbits must be >= 0')
        if value < 0 or (value >> nbits):
            raise ValueError(f'value {value} does not fit in {nbits} bits')
        acc = (self._acc << nbits) | value
        nacc = self._nacc + nbits
        self._nbits += nbits
        if nacc >= 8:
            rem = nacc & 7
            self._buf += (acc >> rem).to_bytes(nacc >> 3, 'big')
            acc &= (1 << rem) - 1
            nacc = rem
        self._acc, self._nacc = acc, nacc

    def write_exp_golomb(self, v, k=0):
        """Order-k Exp-Golomb code of v >= 0."""
        if v < 0:
            raise ValueError('exp-golomb value must be >= 0')
        vp = v + (1 << k)
        self.write(vp, 2 * vp.bit_length() - k - 1)

    @property
    def bit_length(self):
        return self._nbits

    def getvalue(self):
        out = bytes(self._buf)
        if self._nacc:
            out += bytes([self._acc << (8 - self._nacc)])
        return out


class BitReader:
    """MSB-first bit reader; reading past the end raises EOFError."""

    __slots__ = ('_data', '_pos', '_end', '_acc', '_nacc')

    def __init__(self, data):
        self._data = bytes(data)
        self._pos = 0
        self._end = len(self._data)
        self._acc = 0
        self._nacc = 0

    def read(self, nbits):
        if nbits <= 0:
            if nbits == 0:
                return 0
            raise ValueError('nbits must be >= 0')
        nacc = self._nacc
        if nacc < nbits:
            need = (nbits - nacc + 7) >> 3
            pos = self._pos
            if pos + need > self._end:
                raise EOFError('bit stream exhausted')
            self._acc = (self._acc << (need << 3)) | int.from_bytes(self._data[pos:pos + need], 'big')
            self._pos = pos + need
            nacc += need << 3
        nacc -= nbits
        v = self._acc >> nacc
        self._acc &= (1 << nacc) - 1
        self._nacc = nacc
        return v

    def read_exp_golomb(self, k=0):
        z = 0
        while True:
            if self._nacc == 0:
                if self._pos >= self._end:
                    raise EOFError('bit stream exhausted')
                self._acc = self._data[self._pos]; self._pos += 1; self._nacc = 8
            nacc, acc = self._nacc, self._acc
            if acc == 0:
                z += nacc
                self._acc = self._nacc = 0
                continue
            leadz = nacc - acc.bit_length()
            z += leadz
            nacc -= leadz + 1
            self._acc, self._nacc = acc & ((1 << nacc) - 1), nacc
            break
        return ((1 << (z + k)) | self.read(z + k)) - (1 << k)


def bit_length(v):
    """Exact bit length of non-negative int64 values (0 -> 0), vectorised."""
    v = np.asarray(v, dtype=np.int64)
    e = np.frexp(v.astype(np.float64))[1].astype(np.int64)
    e = np.minimum(e, 63)
    pos = v > 0
    e -= (v < (np.int64(1) << np.maximum(e - 1, 0))) & pos             # float rounding can only round v up
    return np.where(pos, e, 0)


def zigzag(x):
    """0, -1, 1, -2, 2, ... -> 0, 1, 2, 3, 4, ..."""
    x = np.asarray(x, dtype=np.int64)
    return (x << 1) ^ (x >> 63)


def unzigzag(u):
    u = np.asarray(u, dtype=np.int64)
    return (u >> 1) ^ -(u & 1)


def entropy_bits(symbols):
    """Zero-order empirical entropy in bits per symbol."""
    s = np.asarray(symbols).ravel()
    if len(s) == 0:
        return 0.0
    _, counts = np.unique(s, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


# ----------------------------------------------------------------------------------------------------------------
# the adaptive context model
# ----------------------------------------------------------------------------------------------------------------
def model_forward(sym, ctx, nsym, init, adapt_rate=ADAPT_RATE):
    """(start, freq) of every symbol under the adaptive per-context model starting from the counts `init` [nctx, nsym]."""
    r = _fast.model_forward(sym, ctx, nsym, init.shape[0], adapt_rate, init)
    if r is not None:
        return r
    scale = RANS_M - nsym
    limit = nsym + (MODEL_INC << (adapt_rate + 5))
    cnts = [np.asarray(init[c], dtype=np.int64).copy() for c in range(init.shape[0])]
    tots = [int(c.sum()) for c in cnts]
    starts, freqs = [], []
    for s, c in zip(sym, ctx):
        cnt = cnts[c]
        tot = tots[c]
        q = (cnt * scale) // tot
        rem = scale - int(q.sum())
        if s == 0:
            starts.append(0)
            freqs.append(int(q[0]) + 1 + rem)
        else:
            starts.append(int(q[:s].sum()) + s + rem)
            freqs.append(int(q[s]) + 1)
        cnt[s] += MODEL_INC
        tot += MODEL_INC
        if tot > limit:
            cnt += 1
            cnt >>= 1
            tot = int(cnt.sum())
        tots[c] = tot
    return starts, freqs


def rans_encode(starts, freqs):
    """rANS encode of (start, freq) pairs (probability scale 2^15) -> bytes (4-byte final state first)."""
    starts = np.asarray(starts, np.int64).tolist(); freqs = np.asarray(freqs, np.int64).tolist()
    x = RANS_L
    out = bytearray(); put = out.append
    unit = (RANS_L >> RANS_SCALE_BITS) << 8
    SB = RANS_SCALE_BITS
    for i in range(len(freqs) - 1, -1, -1):
        f = freqs[i]
        x_max = unit * f
        while x >= x_max:
            put(x & 0xFF); x >>= 8
        x = ((x // f) << SB) + (x % f) + starts[i]
    out.reverse()
    return x.to_bytes(4, 'little') + bytes(out)


class RansDecoder:
    """Symbol-by-symbol rANS decoder: adaptive symbols (`decode(ctx)`, the model of `model_forward`) and explicit
    frequency tables (`decode_freqs`)."""

    def __init__(self, data, init, nsym, adapt_rate=ADAPT_RATE):
        self.data = bytes(data)
        self.x = int.from_bytes(self.data[:4], 'little') if len(self.data) >= 4 else 0
        self.pos = 4
        self.nsym = nsym
        self.scale = RANS_M - nsym
        self.limit = nsym + (MODEL_INC << (adapt_rate + 5))
        self.cnts = [np.asarray(init[c], dtype=np.int64).copy() for c in range(init.shape[0])]
        self.tots = [int(c.sum()) for c in self.cnts]
        self.ar1 = np.arange(1, nsym + 1, dtype=np.int64)

    def _advance(self, slot, start, freq):
        x = freq * (self.x >> RANS_SCALE_BITS) + slot - start
        while x < RANS_L:
            x = (x << 8) | self.data[self.pos]; self.pos += 1
        self.x = x

    def decode(self, c):
        cnt = self.cnts[c]; tot = self.tots[c]
        q = (cnt * self.scale) // tot
        cum = np.cumsum(q); cum += self.ar1
        rem = RANS_M - int(cum[-1])
        slot = self.x & (RANS_M - 1)
        s = int(np.searchsorted(cum, slot - rem, side='right'))
        if s == 0:
            st, f = 0, int(q[0]) + 1 + rem
        else:
            st, f = int(cum[s - 1]) + rem, int(q[s]) + 1
        self._advance(slot, st, f)
        cnt[s] += MODEL_INC; tot += MODEL_INC
        if tot > self.limit:
            cnt += 1; cnt >>= 1; tot = int(cnt.sum())
        self.tots[c] = tot
        return s

    def decode_freqs(self, f, cum):
        """One symbol under the frequency table f [S] (cum = its cumulative sum)."""
        slot = self.x & (RANS_M - 1)
        s = int(np.searchsorted(cum, slot, side='right'))
        fr = int(f[s])
        self._advance(slot, int(cum[s]) - fr, fr)
        return s
