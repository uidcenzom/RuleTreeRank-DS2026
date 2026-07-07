"""
Confronto RTR (PDT) vs RTRwRuleCard su MQ2007 (LETOR 4.0, Fold1).

Protocollo come nel paper DS2026: prime 500 query con >=10 item, split within-query
(train+valid 70% / test 30%), NDCG@10 a |phi| in {1, 2, 4, 6, 10}.
Iperparametri raccomandati dal paper: d_r=5 (albero di r), d_s=4 (PDT), |xi-xj| attive.
Seed fissato anche nel PDT (via random_state del base_regressor) per riproducibilita'.

Risultati salvati incrementalmente in scripts/mq2007_compare.csv.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # repo root importabile anche da Spyder

import time
import numpy as np
import pandas as pd

from RuleTree import RuleTreeRegressor
from ltr_utility import ModelParam
from ltr_utility.dataset import load_by_query_dataset, DatasetName
from ltr_utility.model_selection.evaluation import evaluate
from ruletreerank import (MixedRTR, PairwiseDistanceTree, KNNRegFast,
                          QueryRanker, RuleCardPairwiseDistance)

SEED = 7
K = 10
PHIS = [1, 2, 4, 6, 10]
OUT_CSV = Path(__file__).resolve().parent / "mq2007_compare.csv"

# MQ2007 Fold1 con il protocollo del paper
train, valid, test, train_valid = load_by_query_dataset(
    REPO_ROOT / "datasets", DatasetName.MQ, hold_out=(0.5, 0.2, 0.3))
print(f"train_valid={train_valid} | test={test}", flush=True)

# iperparametri comuni, quelli raccomandati dal paper
COMMON = dict(
    pdt_depth=4, sdt_depth=5, n_neighbors=5,
    feature_concat=False, feature_diff=True, feature_sq_diff=True,
    subsample=0.5, sdt_max_leaf_nodes=None, min_samples_split=2,
    dist_objective="residuals", verbose=False, n_jobs_leaf=1,
)


class SeededMixRTR(MixedRTR):
    """MixedRTR con PDT: come WrapperMixRTR ma con random_state fissato (PDT deterministico)."""
    def __init__(self, **kw):
        super().__init__(
            distance_f=ModelParam(PairwiseDistanceTree, {
                "base_regressor": ModelParam(RuleTreeRegressor, {
                    "max_depth": kw["pdt_depth"], "random_state": SEED}),
                "feature_concat": kw["feature_concat"],
                "feature_diff": kw["feature_diff"],
                "feature_sq_diff": kw["feature_sq_diff"],
                "subsample": kw["subsample"],
                "verbose": kw["verbose"],
            }),
            aggregation_f=ModelParam(KNNRegFast, {"n_neighbors": kw["n_neighbors"], "n_jobs": 1}),
            base_regressor=RuleTreeRegressor(
                max_depth=kw["sdt_depth"],
                max_leaf_nodes=kw["sdt_max_leaf_nodes"],
                min_samples_split=kw["min_samples_split"],
                random_state=SEED),
            dist_objective=kw["dist_objective"],
            verbose=kw["verbose"],
            n_jobs_leaf=kw["n_jobs_leaf"],
        )


class SeededMixRTRRuleCard(MixedRTR):
    """MixedRTR con RuleCard pairwise al posto del PDT (RTRwRuleCard), seed fissato."""
    def __init__(self, **kw):
        super().__init__(
            distance_f=ModelParam(RuleCardPairwiseDistance, {
                "base_regressor": ModelParam(RuleTreeRegressor, {
                    "max_depth": kw["pdt_depth"], "random_state": SEED}),
                "feature_concat": kw["feature_concat"],
                "feature_diff": kw["feature_diff"],
                "feature_sq_diff": kw["feature_sq_diff"],
                "subsample": kw["subsample"],
                "verbose": kw["verbose"],
                "learning_rate": kw.get("rulecard_lr", 0.2),
                "max_n_iter": kw.get("rulecard_max_n_iter", 20),
                "patience": kw.get("rulecard_patience", 3),
            }),
            aggregation_f=ModelParam(KNNRegFast, {"n_neighbors": kw["n_neighbors"], "n_jobs": 1}),
            base_regressor=RuleTreeRegressor(
                max_depth=kw["sdt_depth"],
                max_leaf_nodes=kw["sdt_max_leaf_nodes"],
                min_samples_split=kw["min_samples_split"],
                random_state=SEED),
            dist_objective=kw["dist_objective"],
            verbose=kw["verbose"],
            n_jobs_leaf=kw["n_jobs_leaf"],
        )


rtr_params = dict(COMMON)
rc_params = dict(COMMON, rulecard_lr=0.2, rulecard_max_n_iter=20, rulecard_patience=3)

all_rows = []


def run(name, model_cls, params):
    last_ranker = None
    for phi in PHIS:
        np.random.seed(SEED)
        t0 = time.time()
        ranker = QueryRanker(ranker=ModelParam(model=model_cls, param=params),
                             q_per_model=phi).fit(train=train_valid)
        fit_time = time.time() - t0
        pred = ranker.predict(X=test.x, q=test.q)
        m, s, med = evaluate(pred=pred, labels=test.y, groups_count=test.group_count, k=K)
        print(f"[{name}] |phi|={phi}: NDCG@{K}={m:.4f} (std {s:.3f}) | fit={fit_time:.1f}s", flush=True)
        all_rows.append({"model": name, "phi": phi, f"ndcg@{K}": round(float(m), 4),
                         f"ndcg@{K}_std": round(float(s), 4), "fit_s": round(fit_time, 1)})
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)  # salvataggio incrementale
        last_ranker = ranker
    return last_ranker


print("\n--- RTR (PDT baseline) ---", flush=True)
run("RTR (PDT)", SeededMixRTR, rtr_params)
print("\n--- RTRwRuleCard ---", flush=True)
rc_last = run("RTRwRuleCard", SeededMixRTRRuleCard, rc_params)

df = pd.DataFrame(all_rows)
print("\n================ MQ2007 — CONFRONTO NDCG@10 ================")
print(df.pivot(index="phi", columns="model", values=f"ndcg@{K}").to_string())
print("\n================ MQ2007 — Tempi di training (s) ================")
print(df.pivot(index="phi", columns="model", values="fit_s").to_string())
print(f"\nSaved: {OUT_CSV}")

# esempio qualitativo, scheda additiva RuleCard di una foglia
print("\n================ Esempio scheda RuleCard (una foglia) ================")
cols = [f"f{i+1}" for i in range(train_valid.x.shape[1])]
inner = next(iter(rc_last._models_to_qs.keys()))
for key, agg in inner._leaf_dist_map.items():
    dist = agg.custom_metric_func
    if dist is not None and not getattr(dist, "_fallback", True):
        rules = dist.get_rules(columns_names=cols)
        print(f"Foglia/query {key}: {len(rules)} regole additive (prime 3):")
        for r in rules[:3]:
            print("  su", r["features"], "->", r["rules"])
        break
else:
    print("(nessuna foglia con modello RuleCard non-fallback in questo run)")
