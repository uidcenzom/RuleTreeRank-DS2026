import numpy as np
from numba import njit, prange
from sklearn.metrics import pairwise_distances

@njit(parallel=True, cache=True)
def _build_pairs_numba(Xa, Xb):
    na, da = Xa.shape
    nb, db = Xb.shape
    out = np.empty((na * nb, da + db), dtype=Xa.dtype)

    for i in prange(na):
        row_base = i * nb
        xa_row = Xa[i]
        for j in range(nb):
            row = row_base + j
            out[row, :da] = xa_row
            out[row, da:] = Xb[j]

    return out


def generate_pairwise_dataset(Xa: np.ndarray, Xb: np.ndarray, metric='euclidean', ya=None, yb=None,
                              n_jobs=1):
    X_pairs = _build_pairs_numba(Xa, Xb)

    assert (ya is None and yb is None) or (ya is not None and yb is not None)
    if ya is not None and ya.ndim == 1:
        ya = ya[:, None]
    if yb is not None and yb.ndim == 1:
        yb = yb[:, None]

    y = pairwise_distances(
        Xa if ya is None else ya,
        Xb if yb is None else yb,
        metric=metric,
        n_jobs=n_jobs
    ).ravel()

    return X_pairs, y
