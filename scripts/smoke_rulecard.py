import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root importabile anche da Spyder

import numpy as np
from RuleTree import RuleTreeRegressor
from ltr_utility import ModelParam
from ruletreerank import KNNRegFast, RuleCardPairwiseDistance, RuleTreeRank

rng = np.random.default_rng(0)
X = rng.normal(size=(120, 5))
y = X @ np.array([1.4, -0.7, 0.5, 0.0, 0.2]) + rng.normal(0, 0.1, 120)
q = np.repeat(np.arange(12), 10)

model = RuleTreeRank(
    distance_f=ModelParam(RuleCardPairwiseDistance, {
        "base_regressor": ModelParam(RuleTreeRegressor, {"max_depth": 2, "random_state": 0}),
        "feature_concat": False, "feature_diff": True, "feature_sq_diff": True,
        "subsample": 0.6, "verbose": False,
        "learning_rate": 0.2, "max_n_iter": 25, "patience": 3,
    }),
    aggregation_f=ModelParam(KNNRegFast, {"n_neighbors": 5, "n_jobs": 1}),
    base_regressor=RuleTreeRegressor(max_depth=2, random_state=0),
    dist_objective="residuals",
    verbose=False,
)
model.fit(X, y, q)
print("leaf models:", len(model._leaf_dist_map))
s_full = model.predict(X, q=q, output="full")
s_fast_false = model.predict(X, q=q, output="full", fast=False)
print("full[:5]:", np.round(s_full[:5], 3))
print("fast=False[:5]:", np.round(s_fast_false[:5], 3))
print("finite:", np.isfinite(s_full).all(), "| shape:", s_full.shape)

# show a RuleCard additive-rule sample from one leaf
leaf, agg = next(iter(model._leaf_dist_map.items()))
dist = agg.custom_metric_func
print("distance model type:", type(dist).__name__, "| fallback:", getattr(dist, "_fallback", None))
if dist is not None and hasattr(dist, "get_rules"):
    rules = dist.get_rules(columns_names=[f"f{i}" for i in range(5)])
    print("n additive estimators:", len(rules))
print("SMOKE OK")
