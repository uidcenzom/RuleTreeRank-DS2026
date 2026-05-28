from typing import Generator

import numpy as np


class QueryKFold:
    def __init__(self, n_splits: int = 3):
        if n_splits <= 1:
            raise ValueError("The number of folds must be greater than 1.")
        self.n_splits = n_splits

    def split(self, X: np.ndarray, y: np.ndarray, q: np.ndarray) -> Generator:
        if not (X.shape[0] == y.shape[0] == q.shape[0]):
            raise ValueError("X,y and q must have the same shape.")

        folds_train_idx = [[] for _ in range(self.n_splits)]
        folds_test_idx = [[] for _ in range(self.n_splits)]
        unique_q, inverse_idx = np.unique(q, return_inverse=True)

        for i in range(len(unique_q)):
            q_indices = np.where(inverse_idx == i)[0]
            chunks = np.array_split(q_indices, self.n_splits)

            for fold_idx in range(self.n_splits):
                ts_rel = chunks[fold_idx]
                tr_chunks = [chunks[j] for j in range(self.n_splits) if j != fold_idx]
                tr_rel = np.concatenate(tr_chunks) if tr_chunks else np.array([], dtype=int)

                folds_train_idx[fold_idx].append(tr_rel)
                folds_test_idx[fold_idx].append(ts_rel)

        for f in range(self.n_splits):
            tr_idx = np.concatenate(folds_train_idx[f]).astype(int)
            ts_idx = np.concatenate(folds_test_idx[f]).astype(int)
            yield (X[tr_idx], y[tr_idx], q[tr_idx]), (X[ts_idx], y[ts_idx], q[ts_idx])
