"""Float checkpoint -> the fixed-point model the codec loads.

  python -m curvecodec.train.export CKPT.pt OUT.npz
"""
import hashlib
import sys

import numpy as np


def export(ckpt, out):
    import torch
    from ..model import export_int_model
    ck = torch.load(ckpt, map_location='cpu', weights_only=True)
    export_int_model({k: v.detach().cpu().numpy().astype(np.float64) for k, v in ck['state'].items()}, ck['args'], out)
    return hashlib.sha256(open(out, 'rb').read()).hexdigest()


if __name__ == '__main__':
    print(export(sys.argv[1], sys.argv[2]))
