import math
import time
from datetime import datetime
from pathlib import Path
from typing import Union, Literal, List, Tuple, Dict, Optional

import numpy as np
from numpy import ndarray

from ltr_utility import ModelParam, TreeModel
from sklearn.metrics import euclidean_distances, pairwise_distances


class PairwiseDistanceTree:
    """
    Interpretable regressor that learns to approximate pairwise distances between instances and then uses those learned
    distances to refine local ordering within leaves of an upstream shallow tree (e.g., a rule-based regressor).
    PairwiseDistanceTree constructs a pairwise dataset by combining or differencing feature vectors (ζ transformation),
    supervises on target distances (from labels or residuals), and trains a shallow decision tree to produce symbolic
    and human-readable rules.
    """

    def __init__(self,
                 base_regressor: Optional[ModelParam] = None,
                 feature_concat: bool = False,
                 feature_diff: bool = True,
                 feature_sq_diff: bool = True,
                 subsample: Union[float, int] = 1.,
                 verbose: bool = False):
        """
            Parameters
            ----------
            base_regressor : TreeModel
                An instance of ShallowRegressor specifying the underlying regressor to use for pairwise distance
                approximation. If None, defaults to RuleTreeRegressor.

            feature_concat : bool, default=True
                If True, the pairwise feature representation includes concatenation of the two instances `[xa, xb]`.

            feature_diff : bool, default=True
                If True, the pairwise feature representation includes absolute differences `|xa - xb|` appended to the
                concatenation.

            feature_sq_diff: bool = True
                If True, the pairwise feature representation includes squared differences `(xa - xb)^2` appended to the
                concatenation and absolute differences.

            subsample : float or int, default=1.0
                Controls how many instances from Xa and Xb are used to build the pairwise dataset. If float, interpreted as
                a fraction in (0,1]; if int, the exact count.

        """

        #  The underlying regressor instance (e.g., a shallow decision tree).
        # assert (base_regressor is not None and base_regressor.model == RuleTreeRegressor) or (base_regressor is None), \
        #     "Currently, only RuleTreeRegressor is supported as base_regressor. of PDT."

        if base_regressor is None:
            from RuleTree import RuleTreeRegressor
            self.regressor = RuleTreeRegressor()
            self.regressor_par = {}
            self.random_state = None  # Default
        else:
            self.regressor = base_regressor.model(**base_regressor.param)
            self.regressor_par = base_regressor.param
            self.random_state = base_regressor.param.get('random_state', None)

        self.feature_concat = feature_concat  # Whether to use feature concatenation in the pair representation.
        self.feature_diff = feature_diff  # Whether to append absolute feature differences to the pair representation.
        self.feature_sq_diff = feature_sq_diff  # Whether to append squared feature differences to the pair representation.
        self.subsample = subsample

        self.num_features_: Union[int, None] = None  # Number of features seen during fit
        self.verbose = verbose  # Verbosity flag

        assert (feature_diff | feature_concat | feature_sq_diff), \
            "At least one of 'feature_diff', 'feature_concat' or 'feature_sq_diff' must be True."

    def fit(self, X: ndarray, z: Optional[ndarray] = None, distances: Optional[ndarray] = None,
            mask: Union[ndarray, None, List] = None) -> 'PairwiseDistanceTree':
        """
        Fit the underlying regressor on the pairwise dataset built from Xa and Xb.

        Parameters
        ----------
        X, : ndarray of shape (na, d)
            Blocks of instances to be paired. Must have the same number of features.

        z : ndarray or None
            Optional labels/vectors used to compute pairwise targets (see `_generate_pairwise_dataset`).

        distances: ndarray or None
            Optional precomputed pairwise distance matrix to be used as targets instead of computing distances

        mask: ndarray or None
            Optional mask of instances to consider from X. If None, all instances are used.
        """
        assert X.ndim == 2, "Input X must be a 2D array."
        assert X.shape[0] > 0, "At least one instance are required to build pairwise dataset."

        # If mask are provided, subset X and z accordingly
        assert (z is None) or (z.ndim == 1 and X.shape[0] == z.shape[0]), \
            f"If provided, z must have the same number of instances as X., got: {X.shape, z.shape}."

        # Subset X and z if mask are provided
        assert (distances is None) or (distances.shape == (X.shape[0], X.shape[0])), \
            f"If provided, distances must be a square matrix with shape ({X.shape[0]}, {X.shape[0]}), got: {distances.shape}."

        if mask is not None and isinstance(mask, list):
            mask = np.asarray(mask, dtype=bool)

        assert (mask is None) or (
                mask.ndim == 1 and mask.shape[0] == X.shape[0] and mask.dtype == np.bool_), \
            "mask must be a boolean array with the same length as the number of instances in X."

        if mask is None:
            # We use the slice to avoid the numpy copy that would happen with X[...]
            mask = np.ones(X.shape[0], dtype=bool)

        self.num_features_ = X.shape[1]

        # Remember: X has shape (n_instances, n_features), mask have shape (n_instances, )
        # but is True only for the instances we want to consider i.e. the instance that belong to the leaves we want to refine
        dim_x = mask.sum(dtype=int)  # the number of instances that we can consider (the occurrence of TRUE)
        size = math.ceil(dim_x * self.subsample) if isinstance(self.subsample, float) else min(self.subsample, dim_x)

        assert size <= dim_x, " Subsample size cannot be larger than the number of available instances."

        # Subsample mask with reproducible seed, then we pick exactly "n_instances",
        rng = np.random.default_rng(self.random_state)

        # A list of mask with dimension "size" where for each index 0 <= i < SUM(mask) = (the occurrence of TRUE)
        # For example [0 1 0 1 1 0 1 0 0 1] => random value from 0 to 4 e.g., subset of 3 will be [1,2,4] but not [1,2,5],
        subset_idx_a = rng.choice(dim_x, size, replace=False) if size < dim_x else np.arange(dim_x)
        subset_idx_b = rng.choice(dim_x, size, replace=False) if size < dim_x else np.arange(dim_x)

        # print("subset_idx_a", subset_idx_a, "subset_idx_b", subset_idx_b)

        # A list of mask where mask is TRUE e.g., [1,3,4,6,9] in the previous example
        indices_true = np.flatnonzero(mask)

        indices_target_a = indices_true[subset_idx_a]  # e.g. [1,2,4]
        indices_target_b = indices_true[subset_idx_b]  # e.g. [0,2,3]

        subset_idx_new_a = np.zeros(len(mask), dtype=bool)
        subset_idx_new_a[indices_target_a] = True
        subset_idx_new_b = np.zeros(len(mask), dtype=bool)
        subset_idx_new_b[indices_target_b] = True

        # ---------- Compute x pairs --------------
        start_x_pair = time.time()

        x_pairs = self._generate_pairwise_x_dataset(X, subset_idx_new_a, subset_idx_new_b)

        end_x_pair = time.time()
        # ---------- Compute x pairs --------------

        # ---------- Compute pairwise distances for targets --------------
        if distances is None:

            if self.verbose: print("Log ---", datetime.now(),
                                   "--- Computing pairwise distances for {} instances...".format(X.shape[0]))
            start_distance = time.time()
            # --------------------------
            if z is None:
                xa_sub, xb_sub = X[subset_idx_new_a], X[subset_idx_new_b]
                y_pairs = euclidean_distances(X=xa_sub, Y=xb_sub, squared=True).ravel().astype(np.float32)
                del xa_sub, xb_sub
            else:
                za_sub, zb_sub = z[subset_idx_new_a].reshape(-1, 1), z[subset_idx_new_b].reshape(-1, 1)
                y_pairs = euclidean_distances(X=za_sub, Y=zb_sub, squared=True).ravel().astype(np.float32)
                del za_sub, zb_sub
            # --------------------------
            time_dist_elapsed = round(time.time() - start_distance, 4)
            if self.verbose: print("Log ---", datetime.now(),
                                   "--- End of distance computation in: {}s".format(time_dist_elapsed))
        else:
            if self.verbose: print("Log ---", datetime.now(),
                                   "--- Using provided pairwise distance matrix for {} instances...".format(X.shape[0]))
            time_dist_elapsed = 0
            # --------------------------
            y_pairs = distances[np.ix_(subset_idx_new_a, subset_idx_new_b)].ravel()

        # ---------- Compute pairwise distances for targets --------------

        # ---------- Fit the regressor on pairwise data --------------
        if self.verbose: print("Log ---", datetime.now(),
                               "--- Fitting PairwiseDistanceTree on {} pairs...".format(x_pairs.shape[0]))
        start_regressor = time.time()

        self.regressor.fit(x_pairs, y_pairs)
        end_regressor = time.time()
        if self.verbose: print("Log ---", datetime.now(),
                               "--- End of regressor fitting in: {}s".format(round(end_regressor - start_regressor, 4)))
        # ---------- Fit the regressor on pairwise data --------------
        del x_pairs, y_pairs

        return self

    def _generate_pairwise_x_dataset(self, X: ndarray, indices_a: Optional[ndarray] = None,
                                     indices_b: Optional[ndarray] = None) -> ndarray:
        """
        Generate the pairwise feature dataset X_pairs from the original feature matrix X.

        Parameters
        ----------
        X : ndarray of shape (n_instances, n_features)
            Original feature matrix.
        indices_a, indices_b : ndarray of shape (n_instances,)
            Boolean arrays indicating which instances to consider from X for pairing.

        Returns
        -------
        x_pairs : ndarray of shape (n_pairs, n_pair_features)
            Pairwise feature dataset constructed from X.

        """

        if indices_a is None:
            indices_a = np.ones(X.shape[0], dtype=bool)

        if indices_b is None:
            indices_b = np.ones(X.shape[0], dtype=bool)

        assert X.shape[0] == indices_a.shape[0], "X and indices_a must have the same number of instances."
        assert indices_a.shape == indices_b.shape, "indices_a and indices_b must have the same shape."
        assert indices_a.dtype == indices_b.dtype == np.bool_, "indices_a and indices_b must be boolean arrays."

        dim_a, dim_b = indices_a.sum(dtype=int), indices_b.sum(dtype=int)

        assert dim_a == dim_b, "For Cartesian product, the number of selected instances in indices_a and indices_b must be the same."

        # in a few words, it makes a Cartesian product
        idx_a = np.repeat(np.arange(dim_a), dim_b)
        idx_b = np.tile(np.arange(dim_b), dim_a)

        idx_a_final = np.flatnonzero(indices_a)[idx_a]
        idx_b_final = np.flatnonzero(indices_b)[idx_b]

        parts, tmp_a, tmp_b = [], None, None

        def get_tmps():
            nonlocal tmp_a, tmp_b
            if tmp_a is None:
                tmp_a, tmp_b = X[idx_a_final], X[idx_b_final]
            return tmp_a, tmp_b

        if self.feature_concat:
            tmp_a, tmp_b = get_tmps()
            parts.append(tmp_a)
            parts.append(tmp_b)

        if self.feature_diff:
            tmp_a, tmp_b = get_tmps()
            parts.append(np.abs(tmp_a - tmp_b))

        if self.feature_sq_diff:
            tmp_a, tmp_b = get_tmps()
            parts.append(np.power(tmp_a - tmp_b, 2))

        return np.concatenate(parts, axis=1)

    def _prepare_input(self, x_a: ndarray, x_b: ndarray):
        # TODO: Scrivere documentazione
        parts = []
        if self.feature_concat:
            parts.extend([x_a, x_b])
        if self.feature_diff:
            parts.append(np.abs(x_a - x_b))
        if self.feature_sq_diff:
            parts.append(np.power(x_a - x_b, 2))

        overall_x = np.concatenate(parts, axis=1)
        return overall_x

    def predict(self, x_a: ndarray, x_b: ndarray) -> ndarray:
        """
        Predict pairwise distances between instances in Xa and Xb.

        Parameters
        ----------
        x_a, x_b : ndarray of shape (na, d) and (nb, d)
            Blocks of instances to be paired. Must have the same number of features.

        Returns
        -------
        distances : ndarray of shape (na*nb,)
            Predicted pairwise distances between instances in Xa and Xb.
        """
        assert x_a.shape[1] == self.num_features_, \
            "Input features must match the number of features seen during training."
        assert x_a.shape == x_b.shape, "Xa and Xb must have same shape for direct prediction."

        return self.regressor.predict(self._prepare_input(x_a, x_b))

    def apply(self, x_a: ndarray, x_b: ndarray) -> ndarray:
        # TODO: Scrivere documentazione

        assert x_a.shape[1] == self.num_features_, \
            "Input features must match the number of features seen during training."
        assert x_a.shape == x_b.shape, "Xa and Xb must have same shape for direct prediction."

        return self.regressor.apply(self._prepare_input(x_a, x_b))

    def get_rules(self):
        # TODO: Scrivere documentazione
        return self.regressor.get_rules()

    def construct_feat_dict(self, columns_names: ndarray) -> Dict[int, str]:

        c = 0
        idx2column = {}

        if self.feature_concat:
            for i in range(self.num_features_):
                idx2column[c + i] = "A_" + str(columns_names[i])
            c += self.num_features_
            for i in range(self.num_features_):
                idx2column[c + i] = "B_" + str(columns_names[i])
            c += self.num_features_
        if self.feature_diff:
            for i in range(self.num_features_):
                idx2column[c + i] = f"|A_{str(columns_names[i])} - B_{columns_names[i]}|"
            c += self.num_features_
        if self.feature_sq_diff:
            for i in range(self.num_features_):
                idx2column[c + i] = f"(A_{str(columns_names[i])} - B_{str(columns_names[i])})^2"
            c += self.num_features_

        return idx2column
