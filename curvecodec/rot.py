"""Quaternion helpers (numpy, float64).

Conventions follow RTM / ACL: a quaternion is stored as (x, y, z, w); quat_mul(a, b) applies a, then b (the Hamilton
product b * a); rotate(v, q) = q v q*.
"""
import numpy as np


def quat_mul(a, b):
    """Apply rotation a, then b. Shapes broadcast over the leading dimensions; last dimension 4 (x, y, z, w)."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x = bw * ax + bx * aw + by * az - bz * ay
    y = bw * ay - bx * az + by * aw + bz * ax
    z = bw * az + bx * ay - by * ax + bz * aw
    w = bw * aw - bx * ax - by * ay - bz * az
    return np.stack([x, y, z, w], axis=-1)


def quat_normalize(q):
    n = np.sqrt(np.sum(q * q, axis=-1, keepdims=True))
    return q / np.maximum(n, 1e-30)


def rotate(v, q):
    """q v q* for v [..., 3], q [..., 4]."""
    qv = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_cols(q):
    """Unit quaternions [F, 4] -> the three columns of their rotation matrices [F, 3 (column), 3 (xyz)] (= rotate(e_i, q))."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z
    out = np.empty((len(q), 3, 3))
    out[:, 0, 0] = 1 - 2 * (yy + zz); out[:, 0, 1] = 2 * (xy + wz); out[:, 0, 2] = 2 * (xz - wy)
    out[:, 1, 0] = 2 * (xy - wz); out[:, 1, 1] = 1 - 2 * (xx + zz); out[:, 1, 2] = 2 * (yz + wx)
    out[:, 2, 0] = 2 * (xz + wy); out[:, 2, 1] = 2 * (yz - wx); out[:, 2, 2] = 1 - 2 * (xx + yy)
    return out


def hemisphere_signs(q):
    """[F, 4] -> [F] of +-1 so that q * s[:, None] has non-negative consecutive dot products and frame 0 has w >= 0."""
    d = np.sum(q[1:] * q[:-1], axis=-1)
    s = np.ones(len(q))
    s[1:] = np.where(d < 0.0, -1.0, 1.0)
    s = np.cumprod(s)
    if q[0, 3] < 0.0:
        s = -s
    return s


def log_map(q):
    """Unit quaternions [F, 4] of one track -> rotation vectors [F, 3] (axis x angle) of the hemisphere-continuous track,
    angle = 2 atan2(|xyz|, w) in [0, 2 pi)."""
    qq = q * hemisphere_signs(q)[:, None]
    xyz = qq[:, :3]
    s = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(s, qq[:, 3])
    k = np.where(s > 1e-12, angle / np.maximum(s, 1e-30), 2.0)
    return xyz * k[:, None]


def exp_map(v):
    """Rotation vectors [..., 3] -> unit quaternions [..., 4]."""
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    half = 0.5 * theta
    k = np.where(theta > 1e-12, np.sin(half) / np.maximum(theta, 1e-30), 0.5)
    return np.concatenate([v * k, np.cos(half)], axis=-1)
