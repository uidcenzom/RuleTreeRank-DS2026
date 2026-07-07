from typing import Union

import numpy as np
from RuleTree import RuleTreeRegressor
from lightgbm import LGBMRanker
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.neighbors import KNeighborsRegressor

from ltr_utility import RankerModel, ModelParam, TreeModel
from ltr_utility.dataset import LtrDataset
from ruletreerank import MixedRTR, PairwiseDistanceTree, KNNRegFast, RuleTreeRank, RuleCardPairwiseDistance


class WrapperLGBMRanker(RankerModel):
    def __init__(self, *args, **kwargs):
        kwargs['force_row_wise'] = True
        super().__init__(**kwargs)
        self.model = LGBMRanker(*args, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, q: np.ndarray, *args, **kwargs):
        self.model.fit(X=X, y=y, group=LtrDataset.q2count(q))
        return self

    def predict(self, X: np.ndarray, q: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self.model.predict(X)


class WrapperRuleTree(RankerModel):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self._model = RuleTreeRegressor(*args, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, q: np.ndarray, *args, **kwargs):
        self._model.fit(X=X, y=y)
        return self

    def predict(self, X: np.ndarray, q: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self._model.predict(X)

class RandomRanker(RankerModel):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, q: np.ndarray, *args, **kwargs):
        return self

    def predict(self, X: np.ndarray, q: np.ndarray, *args, **kwargs) -> np.ndarray:
        return np.random.uniform(0,1, size=X.shape[0])


class WrapperRTR(RuleTreeRank):
    def __init__(self, **kwargs):
        super().__init__(
            distance_f=ModelParam(PairwiseDistanceTree, {
                "base_regressor": ModelParam(RuleTreeRegressor, {"max_depth": kwargs["pdt_depth"]}),
                "feature_concat": kwargs["feature_concat"],
                "feature_diff": kwargs["feature_diff"],
                "feature_sq_diff": kwargs["feature_sq_diff"],
                "subsample": kwargs["subsample"],
                "verbose": kwargs["verbose"],
            }),
            aggregation_f=ModelParam(KNNRegFast, {"n_neighbors": kwargs["n_neighbors"], "n_jobs": 1}),
            base_regressor=RuleTreeRegressor(
                max_depth=kwargs["sdt_depth"],
                max_leaf_nodes=kwargs["sdt_max_leaf_nodes"],
                min_samples_split=kwargs["min_samples_split"]),
            dist_objective=kwargs["dist_objective"],
            verbose=kwargs["verbose"],
            n_jobs_leaf=kwargs["n_jobs_leaf"],
        )

class WrapperMixRTR(MixedRTR):
    def __init__(self, **kwargs):
        super().__init__(
            distance_f=ModelParam(PairwiseDistanceTree, {
                "base_regressor": ModelParam(RuleTreeRegressor, {"max_depth": kwargs["pdt_depth"]}),
                "feature_concat": kwargs["feature_concat"],
                "feature_diff": kwargs["feature_diff"],
                "feature_sq_diff": kwargs["feature_sq_diff"],
                "subsample": kwargs["subsample"],
                "verbose": kwargs["verbose"],
            }),
            aggregation_f=ModelParam(KNNRegFast, {"n_neighbors": kwargs["n_neighbors"], "n_jobs": 1}),
            base_regressor=RuleTreeRegressor(
                max_depth=kwargs["sdt_depth"],
                max_leaf_nodes=kwargs["sdt_max_leaf_nodes"],
                min_samples_split=kwargs["min_samples_split"]),
            dist_objective=kwargs["dist_objective"],
            verbose=kwargs["verbose"],
            n_jobs_leaf=kwargs["n_jobs_leaf"],
        )

class WrapperMixRTRRuleCard(MixedRTR):
    """MixedRTR (query-aware) with the PDT replaced by a RuleCard pairwise-distance (RTRwRuleCard)."""
    def __init__(self, **kwargs):
        super().__init__(
            distance_f=ModelParam(RuleCardPairwiseDistance, {
                "base_regressor": ModelParam(RuleTreeRegressor, {"max_depth": kwargs["pdt_depth"]}),
                "feature_concat": kwargs["feature_concat"],
                "feature_diff": kwargs["feature_diff"],
                "feature_sq_diff": kwargs["feature_sq_diff"],
                "subsample": kwargs["subsample"],
                "verbose": kwargs["verbose"],
                "learning_rate": kwargs.get("rulecard_lr", 0.1),
                "max_n_iter": kwargs.get("rulecard_max_n_iter", 50),
                "patience": kwargs.get("rulecard_patience", 5),
            }),
            aggregation_f=ModelParam(KNNRegFast, {"n_neighbors": kwargs["n_neighbors"], "n_jobs": 1}),
            base_regressor=RuleTreeRegressor(
                max_depth=kwargs["sdt_depth"],
                max_leaf_nodes=kwargs["sdt_max_leaf_nodes"],
                min_samples_split=kwargs["min_samples_split"]),
            dist_objective=kwargs["dist_objective"],
            verbose=kwargs["verbose"],
            n_jobs_leaf=kwargs["n_jobs_leaf"],
        )

class WrapperKNN(RankerModel):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.model = KNeighborsRegressor(*args, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, q: np.ndarray, *args, **kwargs):
        self.model.n_neighbors = min(self.model.n_neighbors, X.shape[0])
        self.model.fit(X=X, y=y)
        return self

    def predict(self, X: np.ndarray, q: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self.model.predict(X)


class WrapperKNNPDT(RuleTreeRank):
    def __init__(self, **kwargs):

        class FakeRegressor(TreeModel):
            def fit(self, X: Union[DataFrame, ndarray], y: Union[Series, ndarray]) -> "TreeModel":
                return self
            def predict(self, X: Union[DataFrame, ndarray]) -> ndarray:
                return np.zeros(X.shape[0])
            def apply(self, X: Union[DataFrame, ndarray]) -> ndarray:
                return np.asarray(["A"]*X.shape[0])

        super().__init__(
            distance_f=ModelParam(PairwiseDistanceTree, {
                    "base_regressor": ModelParam(RuleTreeRegressor, {"max_depth": kwargs["pdt_depth"]}),
                    "feature_concat": kwargs["feature_concat"],
                    "feature_diff": kwargs["feature_diff"],
                    "feature_sq_diff": kwargs["feature_sq_diff"],
                    "subsample": kwargs["subsample"],
                    "verbose": kwargs["verbose"],
            }),
            aggregation_f=ModelParam(KNNRegFast, {"n_neighbors": kwargs["n_neighbors"], "n_jobs": 1}),
            base_regressor=FakeRegressor(),
            dist_objective=kwargs["dist_objective"],
            verbose=kwargs["verbose"],
            n_jobs_leaf=1,
        )
