"""BVH skeleton parsing (the header; the motion is read by acl_profile, which also converts it to local transforms)."""
import numpy as np


def parse_header(path):
    """-> (joint names, parents int32 [B], rest offsets float32 [B, 3], channels per joint); end sites are not joints.
    Joint names may contain spaces (the rest of the line)."""
    with open(path, 'r', errors='replace') as f:
        lines = []
        for line in f:
            if line.strip().startswith('MOTION'):
                break
            lines.append(line)
    names, parents, offsets, channels = [], [], [], []
    stack = []                                   # joint index, or -1 for an end site
    for line in lines:
        s = line.strip()
        if not s:
            continue
        tok = s.split()
        if tok[0] in ('ROOT', 'JOINT'):
            names.append(s[len(tok[0]):].strip())
            parents.append(next((x for x in reversed(stack) if x >= 0), -1))
            offsets.append([0.0, 0.0, 0.0]); channels.append([])
            stack.append(len(names) - 1)
        elif tok[0] == 'End':
            stack.append(-1)
        elif tok[0] == '}':
            stack.pop()
        elif tok[0] == 'OFFSET':
            if stack and stack[-1] >= 0:
                offsets[stack[-1]] = [float(tok[1]), float(tok[2]), float(tok[3])]
        elif tok[0] == 'CHANNELS':
            if stack and stack[-1] >= 0:
                channels[stack[-1]] = tok[2:2 + int(tok[1])]
    return names, np.asarray(parents, dtype=np.int32), np.asarray(offsets, dtype=np.float32), channels
