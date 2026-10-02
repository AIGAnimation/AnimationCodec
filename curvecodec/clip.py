"""A clip and its ACL reference."""
from dataclasses import dataclass, field

import numpy as np


def prec_key(p):
    return f'{float(p):g}'


@dataclass
class Clip:
    """local float32 [F, B, 7] raw local transforms (qx, qy, qz, qw, tx, ty, tz; cm), the skeleton (parents [B], rest
    offsets [B, 3] in cm, joint names), the frame rate, and ACL's results per precision (`acl[prec_key(p)]`: compressed
    size and error statistics -- the contract's reference)."""
    name: str
    local: np.ndarray
    parents: np.ndarray
    offsets: np.ndarray
    fps: float
    names: list = field(default_factory=list)
    scale: float = 1.0
    acl: dict = field(default_factory=dict)

    @property
    def F(self):
        return int(self.local.shape[0])

    @property
    def B(self):
        return int(self.local.shape[1])

    def ref(self, p):
        return self.acl[prec_key(p)]
