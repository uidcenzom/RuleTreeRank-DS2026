"""
Analisi di sensibilità sugli iperparametri propri di RuleCard.

Negli esperimenti principali gli iperparametri di RuleCard erano fissi
(learning rate 0.2, massimo 20 iterazioni, patience 3). Qui li facciamo variare
uno alla volta attorno a quella configurazione di partenza, per vedere come
cambiano qualità del ranking (NDCG@10) e tempo di allenamento.

Si lavora sul dataset sintetico, che è veloce: l'obiettivo è misurare la
sensibilità, non cercare la configurazione ottimale (quella sarebbe una model
selection, un'altra cosa). Un solo valore di |phi| basta allo scopo.

I risultati vengono salvati in scripts/rulecard_sensitivity.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import numpy as np
import pandas as pd

from ltr_utility import ModelParam
from ltr_utility.dataset import LtrDataset
from ltr_utility.synthetic import generate_query_synthetic_ltr
from ltr_utility.model_selection.evaluation import evaluate
from ruletreerank import QueryRanker
from experiments.wrappers import WrapperMixRTRRuleCard

RANDOM_STATE = 7
np.random.seed(RANDOM_STATE)

NUM_QUERY, DOC_X_QUERY, FEATURES, NUM_BINS = 48, 40, 10, 5
PHI = 4
K = 10

train, valid, test = generate_query_synthetic_ltr(
    num_query=NUM_QUERY, doc_x_query=DOC_X_QUERY, features=FEATURES,
    num_bins=NUM_BINS, train_size=0.6, valid_size=0.2,
    same_seed=False, random_seed=RANDOM_STATE,
)
train_valid = LtrDataset.concat(train, valid)
print(f"train_valid={train_valid} | test={test}", flush=True)

# Configurazione di partenza: stessi iperparametri di mq2007_compare.py
base = dict(
    pdt_depth=4, sdt_depth=5, n_neighbors=5,
    feature_concat=True, feature_diff=True, feature_sq_diff=False,
    subsample=1.0, sdt_max_leaf_nodes=None, min_samples_split=2,
    dist_objective="dist", verbose=False, n_jobs_leaf=1,
    rulecard_lr=0.2, rulecard_max_n_iter=20, rulecard_patience=3,
)

# variazioni una alla volta rispetto alla configurazione di partenza
configurazioni = [
    ("base", {}),
    ("lr=0.05", {"rulecard_lr": 0.05}),
    ("lr=0.1", {"rulecard_lr": 0.1}),
    ("lr=0.5", {"rulecard_lr": 0.5}),
    ("iter=50", {"rulecard_max_n_iter": 50}),
    ("patience=5", {"rulecard_patience": 5}),
    ("stump_depth=3", {"pdt_depth": 3}),
]

righe = []
for nome, override in configurazioni:
    params = dict(base, **override)
    np.random.seed(RANDOM_STATE)
    t0 = time.time()
    ranker = QueryRanker(ranker=ModelParam(model=WrapperMixRTRRuleCard, param=params),
                         q_per_model=PHI).fit(train=train_valid)
    fit_time = time.time() - t0
    pred = ranker.predict(X=test.x, q=test.q)
    m, s, med = evaluate(pred=pred, labels=test.y, groups_count=test.group_count, k=K)
    print(f"[{nome:>14}] NDCG@{K}={m:.4f} (std {s:.3f}) | fit={fit_time:.1f}s", flush=True)
    righe.append({"config": nome,
                  "lr": params["rulecard_lr"],
                  "max_n_iter": params["rulecard_max_n_iter"],
                  "patience": params["rulecard_patience"],
                  "stump_depth": params["pdt_depth"],
                  f"ndcg@{K}": round(float(m), 4),
                  f"ndcg@{K}_std": round(float(s), 4),
                  "fit_s": round(fit_time, 1)})

df = pd.DataFrame(righe)
print("\nRisultati sensibilità RuleCard (|phi|={}):".format(PHI))
print(df.to_string(index=False))

out = Path(__file__).resolve().parent / "rulecard_sensitivity.csv"
df.to_csv(out, index=False)
print(f"\nSalvato: {out}")
