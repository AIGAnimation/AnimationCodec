"""Release tests (pytest).

  * the golden blobs decode to exactly the golden clips (sha256 of the float32 output), with the C paths and in numpy
    -- decoding is integer arithmetic and IEEE operations in a fixed order, identical on every platform;
  * encode -> decode round trip on a clip: the decoded clip is the encoder's own final search state, and it meets the
    contract (blob bytes may differ across platforms in the last bits: libm / BVH parsing feed the encoder's search);
  * the fixed-point model is the export of the shipped float checkpoint; the shipped look-up tables match libm here;
  * the split file loads and passes its check.
"""
import hashlib
import json
import os

import numpy as np
import pytest

from curvecodec import fast
from curvecodec.clip import Clip
from curvecodec.codec import CurveCodec
from curvecodec.contract import Contract
from curvecodec.metric import error_stats, shell_error

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')
INDEX = json.load(open(os.path.join(GOLDEN, 'index.json')))


def _skeleton(name):
    s = json.load(open(os.path.join(GOLDEN, f'{name}_skeleton.json')))
    return np.asarray(s['parents'], np.int32), np.asarray(s['offsets'], np.float32), float(s['fps'])


def _clip(name):
    d = np.load(os.path.join(GOLDEN, f'{name}.npz'))
    return Clip(name=name, local=d['local'], parents=d['parents'], offsets=d['offsets'], fps=float(d['fps']), acl=INDEX['acl'][name])


@pytest.fixture(scope='module')
def codec():
    return CurveCodec()


@pytest.mark.parametrize('use_c', [True, False], ids=['C', 'numpy'])
@pytest.mark.parametrize('case', INDEX['cases'], ids=[c['blob'][:-4] for c in INDEX['cases']])
def test_golden_decode(codec, case, use_c):
    blob = open(os.path.join(GOLDEN, case['blob']), 'rb').read()
    parents, offsets, fps = _skeleton(case['clip'])
    saved = fast.ENABLED
    fast.ENABLED = use_c
    try:
        out = codec.decode(blob, parents, offsets, fps)
    finally:
        fast.ENABLED = saved
    assert out.dtype == np.float32
    assert hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest() == case['sha256_decoded']


@pytest.mark.parametrize('case', [c for c in INDEX['cases'] if c['clip'] == 'cmu_102_14'], ids=lambda c: c['blob'][:-4])
def test_encode_roundtrip(codec, case):
    clip = _clip('cmu_102_14')
    p = case['precision']; contract = Contract.parse(case['contract'])
    blob = codec.encode(clip, p, contract, case['margin'])
    out = codec.decode(blob, clip.parents, clip.offsets, clip.fps)
    assert np.array_equal(out, codec.last_state)                  # the decoder rebuilds the encoder's verified state
    ok, reasons, _ = contract.check(error_stats(shell_error(clip.local, out, clip.parents), p), clip.ref(p), p)
    assert ok, reasons
    if blob != open(os.path.join(GOLDEN, case['blob']), 'rb').read():
        pytest.skip(f'blob differs from the golden one on this platform ({len(blob)} vs {case["bytes"]} bytes); decode / contract fine')


def test_export_reproduces_shipped_model():
    torch = pytest.importorskip('torch')
    from curvecodec.model import MODEL_FILE, export_int_model
    data = os.path.dirname(MODEL_FILE)
    ck = torch.load(os.path.join(data, 'model_float.pt'), map_location='cpu', weights_only=True)
    out = os.path.join(GOLDEN, '..', '_export_test.npz')
    try:
        export_int_model({k: v.numpy().astype(np.float64) for k, v in ck['state'].items()}, ck['args'], out)
        a = np.load(out); b = np.load(MODEL_FILE)
        assert sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    finally:
        if os.path.exists(out):
            os.unlink(out)


def test_tables_match_libm():
    from curvecodec import features
    t = features.compute_tables()
    for k, v in t.items():
        assert np.array_equal(v, getattr(features, k)), k


def test_split():
    from curvecodec import split
    s = split.load()
    n = s.check()
    assert n['test'] > 0 and n['train'] > n['test']


def test_fast_paths_live():
    """With a C compiler every C path builds and loads (else the numpy fallbacks would hide a broken build)."""
    import shutil
    if not fast.ENABLED or shutil.which('cc') is None:
        pytest.skip('C paths disabled / no compiler')
    st = fast.status()
    assert st['kernels']['loaded'] and st['kernels']['search'], st
    assert st['decoder']['loaded'] and st['tfstep']['loaded'], st
