"""
RuleCard-based pairwise distance for RuleTreeRank (RTRwRuleCard).

Replacement for PairwiseDistanceTree: the pairwise distance target is regressed
by PairwiseRuleCardGAM (an additive boosted RuleCard) instead of a single
RuleTreeRegressor. The interface is the same as the PDT: fit builds the pairwise
dataset and trains the model, predict returns the distances for pairs of rows.
The distance target is the squared euclidean distance, computed over z when RTR
passes it (residuals or labels), over the features otherwise, or taken directly
from the precomputed matrix when given.

It subclasses PairwiseDistanceTree so that the isinstance check inside
RuleTreeRank keeps working and no RTR code needs to change.
"""
import copy
import itertools
from typing import List, Optional, Union

import numpy as np
from numpy import ndarray
from sklearn.metrics import euclidean_distances

from RuleTree import RuleTreeRegressor

from ltr_utility import ModelParam
from ruletreerank.pdt import PairwiseDistanceTree
from PairwiseRuleCard.PairwiseRuleCardGAM import PairwiseRuleCardGAM


class _CopyableRuleTreeRegressor(RuleTreeRegressor):
    """RuleTreeRegressor that survives deepcopy.

    This RuleTree version stores an itertools.count as tiebreaker, which cannot be
    copied, while the GAM deep-copies its base estimator at every boosting step.
    A fresh counter on copy is fine because it only breaks ties within a single fit.
    """

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            new.__dict__[k] = itertools.count() if k == "tiebreaker" else copy.deepcopy(v, memo)
        return new


class RuleCardPairwiseDistance(PairwiseDistanceTree):
    """Interpretable pairwise-distance regressor backed by an additive RuleCard (GAM)."""

    def __init__(self,
                 base_regressor: Optional[Union[ModelParam, RuleTreeRegressor]] = None,
                 feature_concat: bool = False,
                 feature_diff: bool = True,
                 feature_sq_diff: bool = False,
                 subsample: Union[float, int] = 1.0,
                 verbose: bool = False,
                 learning_rate: float = 0.1,
                 max_n_iter: int = 50,
                 patience: int = 5,
                 metric: str = "euclidean",
                 random_state: Optional[int] = None):
        # super().__init__() is not called on purpose: the PDT would build a
        # RuleTreeRegressor that is never used here.
        self.feature_concat = feature_concat
        self.feature_diff = feature_diff
        self.feature_sq_diff = feature_sq_diff
        if feature_sq_diff and verbose:
            print("RuleCardPairwiseDistance: feature_sq_diff is unsupported by RuleCard and is ignored.")

        # feature_concat maps to use_pairwise, feature_diff maps to use_difference
        self._use_pairwise = bool(feature_concat)
        self._use_difference = bool(feature_diff)
        if not self._use_pairwise and not self._use_difference:
            # RuleCard needs at least one pair representation
            self._use_difference = True

        self.subsample = subsample
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.max_n_iter = max_n_iter
        self.patience = patience
        self.metric = metric

        # kept as a spec and instantiated at every fit to avoid shared state
        self._base_regressor = base_regressor
        if isinstance(base_regressor, ModelParam):
            self.random_state = base_regressor.param.get("random_state", random_state)
        elif base_regressor is not None:
            self.random_state = getattr(base_regressor, "random_state", random_state)
        else:
            self.random_state = random_state

        self.num_features_: Optional[int] = None
        self.gam_: Optional[PairwiseRuleCardGAM] = None
        self._offset_: float = 0.0
        self._fallback: bool = False  # when True predict uses the plain squared euclidean distance

    def _make_base_estimator(self) -> RuleTreeRegressor:
        br = self._base_regressor
        if isinstance(br, ModelParam):
            if isinstance(br.model, type) and issubclass(br.model, RuleTreeRegressor):
                return _CopyableRuleTreeRegressor(**br.param)
            return br.model(**br.param)
        if isinstance(br, RuleTreeRegressor):
            return _CopyableRuleTreeRegressor(**br.get_params())
        if br is not None:
            return copy.deepcopy(br)
        return _CopyableRuleTreeRegressor(max_depth=3)

    def _pair_target(self, Xm: ndarray, z: Optional[ndarray],
                     distances: Optional[ndarray], mask: ndarray, n_full: int) -> ndarray:
        """Build the squared distance target matrix, same semantics as the PDT."""
        if distances is not None:
            D = np.asarray(distances)
            if D.shape[0] == n_full:
                # full matrix given, keep only the masked block
                D = D[np.ix_(mask, mask)]
            return D
        if z is not None:
            zc = np.asarray(z)[mask].reshape(-1, 1)
            return euclidean_distances(zc, squared=True)
        return euclidean_distances(Xm, squared=True)

    def _raw_predict(self, x_a: ndarray, x_b: ndarray) -> ndarray:
        """Additive RuleCard score, without the batch dependent offset of GAM.predict."""
        g = self.gam_
        x_a = np.asarray(x_a, dtype=float)
        x_b = np.asarray(x_b, dtype=float)
        X_pairs = np.hstack([x_a, x_b])
        if g.use_difference:
            X_diff = np.abs(x_a - x_b)
            X = np.hstack([X_pairs, X_diff]) if g.use_pairwise else X_diff
        else:
            X = X_pairs
        pred = np.ones((X.shape[0],)) * g.base_prediction_
        for feat_idx, est in g.estimators_:
            pred += g.learning_rate * est.predict(X[:, feat_idx].reshape(X.shape[0], -1))
        return pred

    def _compute_offset(self, Xm: ndarray) -> None:
        """Fixed additive constant that keeps distances non negative without changing the kNN ordering."""
        n = Xm.shape[0]
        if n < 2:
            self._offset_ = 0.0
            return
        rng = np.random.default_rng(0 if self.random_state is None else self.random_state)
        m = int(min(2000, n * n))
        ia = rng.integers(0, n, m)
        ib = rng.integers(0, n, m)
        raw = self._raw_predict(Xm[ia], Xm[ib])
        self._offset_ = float(max(0.0, -np.min(raw))) if raw.size else 0.0

    def fit(self, X: ndarray, z: Optional[ndarray] = None,
            distances: Optional[ndarray] = None,
            mask: Union[ndarray, None, List] = None) -> "RuleCardPairwiseDistance":
        X = np.asarray(X)
        assert X.ndim == 2, "X must be a 2D array."

        if mask is None:
            mask = np.ones(X.shape[0], dtype=bool)
        elif isinstance(mask, list):
            mask = np.asarray(mask, dtype=bool)

        self.num_features_ = X.shape[1]
        Xm = X[mask]

        # too few instances for a meaningful pairwise model, use the euclidean fallback
        if Xm.shape[0] < 3:
            self._fallback = True
            self.gam_ = None
            return self

        y_pairs = self._pair_target(Xm, z, distances, mask, X.shape[0])

        base_est = self._make_base_estimator()
        self.gam_ = PairwiseRuleCardGAM(
            learning_rate=self.learning_rate,
            patience=self.patience,
            max_n_iter=self.max_n_iter,
            base_estimator=base_est,
            use_pairwise=self._use_pairwise,
            use_difference=self._use_difference,
            metric=self.metric,
            subsample=self.subsample,
            subsample_strategy="random",
            fast=None,  # avoids the optional interpret dependency
            n_jobs=1,
            random_state=42 if self.random_state is None else int(self.random_state),
            verbose=self.verbose,
        )

        try:
            self.gam_.fit(Xm, mode="precomputed", pair_targets=y_pairs)
        except Exception as exc:
            # if the GAM fit fails this leaf degrades to the euclidean distance
            if self.verbose:
                print(f"RuleCardPairwiseDistance: GAM fit failed ({exc!r}); using euclidean fallback.")
            self._fallback = True
            self.gam_ = None
            return self

        # no boosting round improved, the model is constant, fall back for a useful kNN ordering
        if not getattr(self.gam_, "estimators_", None):
            self._fallback = True
            self.gam_ = None
            return self

        self._fallback = False
        self._compute_offset(Xm)
        return self

    def predict(self, x_a: ndarray, x_b: ndarray) -> ndarray:
        x_a = np.asarray(x_a, dtype=float)
        x_b = np.asarray(x_b, dtype=float)
        assert x_a.shape == x_b.shape, "x_a and x_b must have the same shape."
        assert x_a.shape[1] == self.num_features_, \
            "Input features must match the number of features seen during training."

        if self._fallback or self.gam_ is None:
            # squared euclidean, same as the PDT target
            return np.sum((x_a - x_b) ** 2, axis=1)

        return self._raw_predict(x_a, x_b) + self._offset_

    def get_rules(self, columns_names: Optional[List] = None) -> List[dict]:
        """Additive rules: one shallow RuleTreeRegressor per boosting round."""
        if self.gam_ is None or not getattr(self.gam_, "estimators_", None):
            return []

        nf = self.num_features_
        if columns_names is None:
            columns_names = [f"feat{i}" for i in range(nf)]

        labels: List[str] = []
        if self.gam_.use_pairwise:
            labels += [f"ist1 {c}" for c in columns_names]
            labels += [f"ist2 {c}" for c in columns_names]
        if self.gam_.use_difference:
            labels += [f"diff({c})" for c in columns_names]

        out = []
        for feat_idx, est in self.gam_.estimators_:
            sub = [labels[i] for i in feat_idx]
            try:
                rules = est.get_rules(columns_names=sub)
            except TypeError:
                rules = est.get_rules()
            out.append({"features": sub, "rules": rules})
        return out

    def construct_feat_dict(self, columns_names: ndarray) -> dict:
        nf = self.num_features_
        d, c = {}, 0
        if self.gam_ is not None and self.gam_.use_pairwise:
            for i in range(nf):
                d[c + i] = "A_" + str(columns_names[i])
            c += nf
            for i in range(nf):
                d[c + i] = "B_" + str(columns_names[i])
            c += nf
        if self.gam_ is None or self.gam_.use_difference:
            for i in range(nf):
                d[c + i] = f"|A_{columns_names[i]} - B_{columns_names[i]}|"
            c += nf
        return d
