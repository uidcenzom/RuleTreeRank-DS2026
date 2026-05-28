from typing import Optional, Literal, Tuple, Any

from numpy import argpartition, repeat, tile, arange, full, mean, zeros, float64, ndarray
from sklearn.neighbors import KNeighborsRegressor


class KNNRegFast(KNeighborsRegressor):

    def __init__(self, n_neighbors=5, *, weights='uniform', algorithm='auto',
                 leaf_size=30, p=2, metric='minkowski', metric_params=None,
                 n_jobs:int=None, max_pairs_per_batch:int=2_000_000):

        super().__init__(n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs)

        # Max number of pairs to process per batch in PDT prediction
        self.max_pairs_per_batch = max_pairs_per_batch
        self._custom_metric_func: Optional[Any] = None

    def set_custom_metric(self, f: Any):
        """
        Sets a custom metric function.
        """
        self._custom_metric_func = f

    def predict_slow(self, X: ndarray, method: Literal["default", "euclidian"]):
        assert method in ["default", "euclidian"], "method must be 'default' or 'euclidian'."

        if method == "default":
            return self.predict(X)
        else:
            return (
                KNeighborsRegressor(n_neighbors=self.n_neighbors, metric='euclidean')
                .fit(self._fit_X, self._y)
                .predict(X)
            )


    def predict_fast(self, x: ndarray) -> ndarray:
        """
        Fast implementation of kNN residual correction using PDT distances.

        Parameters
        ----------
        x : ndarray
            Query samples for which to predict residuals.
        knn : KNeighborsRegressor
            kNN regressor trained on residuals within the leaf.
        pdt : PairwiseDistanceTree
            PDT model used as the distance metric for kNN.

        Returns
        ----------
            knn residuals prediction for X.
        """

        if self._custom_metric_func is None: raise ValueError("Custom metric function not set.")

        X_train, k = self._fit_X, self.n_neighbors
        n_query, n_train = x.shape[0], X_train.shape[0]

        if k >= n_train: return full(n_query, mean(self._y))

        batch_size = max(1, self.max_pairs_per_batch // n_train)
        result = zeros(n_query, dtype=float64)

        for i in range(0, n_query, batch_size):
            end_i = min(i + batch_size, n_query)
            current_batch_len = end_i - i

            sub_x = x[i:end_i]

            # Broadcasting
            idx_q = repeat(arange(current_batch_len), n_train)
            idx_t = tile(arange(n_train), current_batch_len)

            dists_batch = self._custom_metric_func.predict(sub_x[idx_q], X_train[idx_t]).reshape(current_batch_len,
                                                                                                 n_train)
            del idx_q, idx_t

            top_k_indices = argpartition(dists_batch, kth=k - 1, axis=1)[:, :k]
            result[i:end_i] = mean(self._y[top_k_indices], axis=1)

            del dists_batch, top_k_indices, sub_x

        return result

    def knn_neighbors_fast(self, x: ndarray):

        if self._custom_metric_func is None: raise ValueError("Custom metric function not set.")

        X_train, k = self._fit_X, self.n_neighbors
        y_train = self._y.ravel() if self._y.ndim > 1 else self._y
        n_query, n_train = x.shape[0], X_train.shape[0]

        if k >= n_train: return full(n_query, mean(y_train))

        batch_size = max(1, self.max_pairs_per_batch // n_train)
        result = zeros((n_query, k), dtype=float64)

        for i in range(0, n_query, batch_size):
            end_i = min(i + batch_size, n_query)
            current_batch_len = end_i - i

            sub_x = x[i:end_i]

            # Broadcasting
            idx_q = repeat(arange(current_batch_len), n_train)
            idx_t = tile(arange(n_train), current_batch_len)

            dists_batch = self._custom_metric_func.predict(
                sub_x[idx_q], X_train[idx_t]).reshape(current_batch_len, n_train)

            del idx_q, idx_t

            top_k_indices = argpartition(dists_batch, kth=k - 1, axis=1)[:, :k]
            result[i:end_i] = top_k_indices

            del dists_batch, top_k_indices, sub_x

        return result

    def get_kneighbors(self, X: ndarray) -> Tuple:

        near_index = self.kneighbors(X)[1][0]
        return self._fit_X[near_index], self._y[near_index]


    @property
    def custom_metric_func(self) -> Any:
        return self._custom_metric_func

