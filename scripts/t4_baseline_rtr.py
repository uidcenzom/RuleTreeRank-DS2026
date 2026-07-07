"""
T4 - Baseline RTR (MixedRTR) su dataset sintetico self-contained.
Riproduce il meccanismo degli esperimenti: QueryRanker(q_per_model=|phi|) + WrapperMixRTR.
Metrica: NDCG@10 (media sulle query del test) via ltr_utility evaluate. Salva anche i tempi di training.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root importabile anche da Spyder

import time
import numpy as np
import pandas as pd

from ltr_utility import ModelParam
from ltr_utility.dataset import LtrDataset
from ltr_utility.synthetic import generate_query_synthetic_ltr
from ltr_utility.model_selection.evaluation import evaluate
from ruletreerank import QueryRanker
from experiments.wrappers import WrapperMixRTR

RANDOM_STATE = 7
np.random.seed(RANDOM_STATE)

# dataset sintetico, numero di query divisibile per 1, 2, 4 e 6
NUM_QUERY = 48
DOC_X_QUERY = 40
FEATURES = 10
NUM_BINS = 5

train, valid, test = generate_query_synthetic_ltr(
    num_query=NUM_QUERY, doc_x_query=DOC_X_QUERY, features=FEATURES,
    num_bins=NUM_BINS, train_size=0.6, valid_size=0.2,
    same_seed=False, random_seed=RANDOM_STATE,
)
train_valid = LtrDataset.concat(train, valid)

print(f"train_valid: {train_valid}  | test: {test}")
print(f"unique queries train_valid={len(train_valid.unique_q)}, test={len(test.unique_q)}")
print(f"relevance labels: {sorted(np.unique(test.y).tolist())}")

# configurazione RTR analoga a quella usata negli esperimenti
rtr_params = {
    "pdt_depth": 2,
    "feature_concat": False,
    "feature_diff": True,
    "feature_sq_diff": True,
    "subsample": 0.5,
    "verbose": False,
    "n_neighbors": 5,
    "sdt_depth": 3,
    "sdt_max_leaf_nodes": None,
    "min_samples_split": 2,
    "dist_objective": "residuals",
    "n_jobs_leaf": 1,
}

PHIS = [1, 2, 4, 6]
K = 10
rows = []

for phi in PHIS:
    print(f"\n===== |phi| = {phi} =====")
    t0 = time.time()
    ranker = QueryRanker(
        ranker=ModelParam(model=WrapperMixRTR, param=rtr_params),
        q_per_model=phi,
    ).fit(train=train_valid)
    fit_time = time.time() - t0

    pred_test = ranker.predict(X=test.x, q=test.q)
    ndcg_mean, ndcg_std, ndcg_med = evaluate(
        pred=pred_test, labels=test.y, groups_count=test.group_count, k=K,
    )
    print(f"|phi|={phi}: NDCG@{K} mean={ndcg_mean:.4f} std={ndcg_std:.4f} median={ndcg_med:.4f} | fit={fit_time:.2f}s")
    rows.append({
        "phi": phi,
        f"ndcg@{K}_mean": round(float(ndcg_mean), 4),
        f"ndcg@{K}_std": round(float(ndcg_std), 4),
        f"ndcg@{K}_median": round(float(ndcg_med), 4),
        "n_models": len(set(id(m) for m in ranker._query_to_model.values())),
        "fit_time_s": round(fit_time, 2),
    })

df = pd.DataFrame(rows)
print("\n===== BASELINE RTR (MixedRTR) — NDCG@10 =====")
print(df.to_string(index=False))

out = Path(__file__).resolve().parent / "t4_baseline_rtr.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
