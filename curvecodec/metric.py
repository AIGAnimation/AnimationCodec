"""Skeleton helpers and ACL's shell error metric.

Local transforms are float arrays [F, B, 7] = (qx, qy, qz, qw, tx, ty, tz) in cm; `parents[b] < b` (-1 for a root).
The error of a bone at a frame is ACL's qvvf_transform_error_metric as cpp/acl_profile.cpp uses it: three virtual
vertices `shell` cm along the bone's local axes, transformed to object space by the raw and by the decoded transforms;
the error is the largest of the three displacements.
"""
import numpy as np

from .rot import quat_mul, quat_normalize, rotate

SHELL_CM = 3.0


def check_parents(parents):
    parents = np.asarray(parents)
    for b, p in enumerate(parents):
        if p >= b:
            raise ValueError(f'parent {p} of bone {b} must precede it')
    return parents


def bone_depth(parents):
    depth = np.zeros(len(parents), dtype=np.int64)
    for b, p in enumerate(np.asarray(parents)):
        depth[b] = 0 if p < 0 else depth[p] + 1
    return depth


def depth_class(depth):
    """0 for the root, 1 for depth 1-2, 2 for depth 3-5, 3 below."""
    depth = np.asarray(depth, dtype=np.int64)
    return sum((depth >= e).astype(np.int64) for e in (1, 3, 6))


def subtree_lists(parents):
    """For every bone the sorted list of the bones of its subtree (itself included)."""
    parents = np.asarray(parents)
    subs = [[b] for b in range(len(parents))]
    for b in range(len(parents) - 1, -1, -1):
        p = int(parents[b])
        if p >= 0:
            subs[p].extend(subs[b])
    return [sorted(s) for s in subs]


def fk_object(local, parents):
    """local [F, B, 7] -> (object rotations [F, B, 4], object positions [F, B, 3]) in float64."""
    local = np.asarray(local, dtype=np.float64)
    parents = check_parents(parents)
    F, B, _ = local.shape
    rot = np.empty((F, B, 4)); pos = np.empty((F, B, 3))
    for b in range(B):
        lq = quat_normalize(local[:, b, :4]); lt = local[:, b, 4:7]
        p = int(parents[b])
        if p < 0:
            rot[:, b] = lq; pos[:, b] = lt
        else:
            rot[:, b] = quat_normalize(quat_mul(lq, rot[:, p]))
            pos[:, b] = rotate(lt, rot[:, p]) + pos[:, p]
    return rot, pos


def shell_error(raw_local, lossy_local, parents, shell=SHELL_CM):
    """Shell error [F, B] in cm of a decoded clip against the raw clip."""
    rr, rp = fk_object(raw_local, parents)
    lr, lp = fk_object(lossy_local, parents)
    err = np.zeros(rr.shape[:2])
    for axis in range(3):
        v = np.zeros(3); v[axis] = shell
        d = np.linalg.norm((rotate(v, rr) + rp) - (rotate(v, lr) + lp), axis=-1)
        err = np.maximum(err, d)
    return err


def error_stats(err, precision):
    """Summary of an error field [F, B]: max / mean / percentiles, fraction above the precision, per-bone max and mean."""
    flat = err.reshape(-1)
    s = np.sort(flat)
    n = len(s)

    def q(p):
        return float(s[min(n - 1, int(p * n))]) if n else 0.0
    return dict(err_max=float(flat.max()) if n else 0.0, err_mean=float(flat.mean()) if n else 0.0,
                err_p50=q(0.5), err_p90=q(0.9), err_p99=q(0.99),
                frac_above=float((flat > precision).mean()) if n else 0.0,
                bone_max=err.max(axis=0).tolist(), bone_mean=err.mean(axis=0).tolist())


def shell_radii(local, parents, shell=SHELL_CM):
    """ACL's dominant shell radius per bone (compute_clip_shell_distances): every bone's 3 cm shell, enlarged leaves
    first so that it encloses the shells of its descendants, R[parent] = max(R[parent], R[b] + max_f |t_b|). A rotation
    error of theta at bone b moves some shell vertex by about R[b] theta."""
    local = np.asarray(local, dtype=np.float64)
    parents = check_parents(parents)
    B = local.shape[1]
    R = np.full(B, float(shell))
    _, obj_pos = fk_object(local, parents)
    max_dist = np.zeros(B)
    for b in range(B):
        p = int(parents[b])
        parent_pos = obj_pos[:, p] if p >= 0 else 0.0
        max_dist[b] = float(np.max(np.linalg.norm(obj_pos[:, b] - parent_pos, axis=-1)))
    for b in range(B - 1, -1, -1):
        p = int(parents[b])
        if p >= 0 and R[b] + max_dist[b] > R[p]:
            R[p] = R[b] + max_dist[b]
    return R
