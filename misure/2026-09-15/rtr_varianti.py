import sys, time
REPO = r"C:\Users\manzo\Desktop\fork\RuleTreeRank-DS2026"
sys.path.insert(0, REPO)
from pathlib import Path
import numpy as np
from ltr_utility import ModelParam
from ltr_utility.dataset import load_by_query_dataset, DatasetName
from ltr_utility.model_selection.evaluation import evaluate
from ruletreerank import QueryRanker
from experiments.wrappers import WrapperMixRTR

tr, va, te, tv = load_by_query_dataset(Path(REPO) / "datasets", DatasetName.MQ, hold_out=(0.5, 0.2, 0.3))
P = dict(pdt_depth=4, feature_concat=True, feature_diff=True, feature_sq_diff=False, subsample=1.0, verbose=False,
         n_neighbors=5, sdt_depth=5, sdt_max_leaf_nodes=None, min_samples_split=2, dist_objective="dist", n_jobs_leaf=1)
Xte, qte = np.asarray(te.x), np.asarray(te.q)

for phi in [1, 10]:
    np.random.seed(7)
    t0 = time.time()
    rk = QueryRanker(ranker=ModelParam(model=WrapperMixRTR, param=P), q_per_model=phi).fit(train=tv)
    fit_s = time.time() - t0
    out = {}
    for mode in ["full", "euclidian", "score"]:
        pred = np.zeros(len(qte))
        for model, qs in rk._models_to_qs.items():
            mk = np.isin(qte, qs)
            if mk.any():
                pred[mk] = model.predict(Xte[mk], q=qte[mk], output=mode)
        m, s, med = evaluate(pred=pred, labels=te.y, groups_count=te.group_count, k=10)
        out[mode] = m
    print(f"phi={phi} | fit {fit_s:.1f}s | NDCG@10 PDT {out['full']:.4f} | euclidea {out['euclidian']:.4f} | solo r(x) {out['score']:.4f}", flush=True)
