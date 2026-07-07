"""
T6 + T7 - RTRwRuleCard vs RTR (PDT) sullo stesso dataset sintetico (uguale a T4).
Confronto NDCG@10 + tempi di training a |phi| in {1,2,4,6}, e campione qualitativo di regole.
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
from experiments.wrappers import WrapperMixRTR, WrapperMixRTRRuleCard

RANDOM_STATE = 7
np.random.seed(RANDOM_STATE)

NUM_QUERY, DOC_X_QUERY, FEATURES, NUM_BINS = 48, 40, 10, 5
train, valid, test = generate_query_synthetic_ltr(
    num_query=NUM_QUERY, doc_x_query=DOC_X_QUERY, features=FEATURES,
    num_bins=NUM_BINS, train_size=0.6, valid_size=0.2,
    same_seed=False, random_seed=RANDOM_STATE,
)
train_valid = LtrDataset.concat(train, valid)
print(f"train_valid={train_valid} | test={test}")

common = dict(
    pdt_depth=2, feature_concat=False, feature_diff=True, feature_sq_diff=True,
    subsample=0.5, verbose=False, n_neighbors=5,
    sdt_depth=3, sdt_max_leaf_nodes=None, min_samples_split=2,
    dist_objective="residuals", n_jobs_leaf=1,
)
rtr_params = dict(common)
rc_params = dict(common)
rc_params.update(rulecard_lr=0.2, rulecard_max_n_iter=20, rulecard_patience=3)

PHIS = [1, 2, 4, 6]
K = 10


def run(name, model_cls, params):
    rows = []
    for phi in PHIS:
        t0 = time.time()
        ranker = QueryRanker(ranker=ModelParam(model=model_cls, param=params),
                             q_per_model=phi).fit(train=train_valid)
        fit_time = time.time() - t0
        pred = ranker.predict(X=test.x, q=test.q)
        m, s, med = evaluate(pred=pred, labels=test.y, groups_count=test.group_count, k=K)
        print(f"[{name}] |phi|={phi}: NDCG@{K}={m:.4f} (std {s:.3f}) | fit={fit_time:.1f}s")
        rows.append({"model": name, "phi": phi, f"ndcg@{K}": round(float(m), 4),
                     f"ndcg@{K}_std": round(float(s), 4), "fit_s": round(fit_time, 1)})
    return rows, ranker  # return last ranker for qualitative inspection


print("\n--- RTR (PDT baseline) ---")
rtr_rows, _ = run("RTR (PDT)", WrapperMixRTR, rtr_params)
print("\n--- RTRwRuleCard ---")
rc_rows, rc_last = run("RTRwRuleCard", WrapperMixRTRRuleCard, rc_params)

df = pd.DataFrame(rtr_rows + rc_rows)
piv = df.pivot(index="phi", columns="model", values=f"ndcg@{K}")
piv_t = df.pivot(index="phi", columns="model", values="fit_s")

print("\n================ T7 — CONFRONTO NDCG@10 ================")
print(piv.to_string())
print("\n================ T7 — Tempi di training (s) ================")
print(piv_t.to_string())

out = Path(__file__).resolve().parent / "t7_compare.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")

# esempio qualitativo, regole additive RuleCard di una foglia
print("\n================ T7 — Leggibilità: esempio scheda RuleCard (una foglia) ================")
cols = [f"f{i}" for i in range(FEATURES)]
shown = 0
# rc_last is a QueryRanker; drill into one fitted WrapperMixRTRRuleCard -> one leaf distance model
inner = next(iter(rc_last._models_to_qs.keys()))  # a fitted model
leaf_map = inner._leaf_dist_map
for key, agg in leaf_map.items():
    dist = agg.custom_metric_func
    if dist is not None and not getattr(dist, "_fallback", True):
        rules = dist.get_rules(columns_names=cols)
        print(f"Foglia/query {key}: {len(rules)} regole additive (mostro le prime 3):")
        for r in rules[:3]:
            print("  su", r["features"], "->", r["rules"])
        shown += 1
        if shown >= 1:
            break
if shown == 0:
    print("(nessuna foglia con modello RuleCard non-fallback in questo run)")
