"""Small numeric helpers shared by data.py and synthesize.py."""
import numpy as np


def minmax_norm(S, min_val, max_val):
    """Scale to [0, 1] between min_val and max_val, clipped."""
    return np.clip((S - min_val) / (max_val - min_val), 0., 1.)


def inv_minmax_norm(x, min_val, max_val):
    """Inverse of minmax_norm for values in [0, 1]."""
    return np.clip(x, 0, 1) * (max_val - min_val) + min_val


def arange_list(a):
    """For run lengths a, returns forward and reverse 1-based positions inside each run.
    Example: [3, 2] -> fwd [1,2,3,1,2], rev [3,2,1,2,1]."""
    fwd = np.array([], dtype=np.uint8)
    rev = np.array([], dtype=np.uint8)
    for x in a:
        idx = np.arange(1, x + 1)
        fwd = np.append(fwd, idx)
        rev = np.append(rev, np.flip(idx))
    return fwd, rev
