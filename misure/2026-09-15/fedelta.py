import sys, time
REPO = r"C:\Users\manzo\Desktop\fork\RuleTreeRank-DS2026"
sys.path.insert(0, REPO)
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import euclidean_distances
from RuleTree import RuleTreeRegressor
from ltr_utility import ModelParam
from ltr_utility.dataset import load_by_query_dataset, DatasetName
from ruletreerank import PairwiseDistanceTree, RuleCardPairwiseDistance

tr, va, te, tv = load_by_query_dataset(Path(REPO) / "datasets", DatasetName.MQ, hold_out=(0.5, 0.2, 0.3))
Xtr, ytr, qtr = np.asarray(tv.x), np.asarray(tv.y), np.asarray(tv.q)
Xte, qte = np.asarray(te.x), np.asarray(te.q)
K = 5
comune = dict(base_regressor=ModelParam(RuleTreeRegressor, {"max_depth": 4}), feature_concat=True,
              feature_diff=True, feature_sq_diff=False, subsample=1.0, verbose=False)
modelli = {
    "PDT": lambda: PairwiseDistanceTree(**comune),
    "RuleCard": lambda: RuleCardPairwiseDistance(**comune, learning_rate=0.2, max_n_iter=20, patience=3),
}
ris = {k: {"rho": [], "overlap": [], "distinti": [], "t": 0.0} for k in modelli}
ris["euclidea_ties"] = []
n_celle = 0
for q in list(tv.unique_q)[:60]:
    mtr, mte = qtr == q, qte == q
    tree = RuleTreeRegressor(max_depth=5, min_samples_split=2, random_state=7).fit(Xtr[mtr], ytr[mtr])
    lf_tr, lf_te = np.asarray(tree.apply(Xtr[mtr])), np.asarray(tree.apply(Xte[mte]))
    Xq, Xqt = Xtr[mtr], Xte[mte]
    for L in np.unique(lf_tr):
        C = Xq[lf_tr == L]
        T = Xqt[lf_te == L]
        if C.shape[0] <= K or T.shape[0] == 0:
            continue
        n_celle += 1
        vero_tt = euclidean_distances(T, C, squared=True)
        top_vero = np.argsort(vero_tt, axis=1)[:, :K]
        ia, ib = np.triu_indices(C.shape[0], 1)
        vero_cc = euclidean_distances(C, squared=True)[ia, ib]
        for nome, crea in modelli.items():
            d = crea()
            t0 = time.time()
            d.fit(C)
            ris[nome]["t"] += time.time() - t0
            pred_cc = d.predict(C[ia], C[ib])
            ris[nome]["rho"].append(spearmanr(pred_cc, vero_cc).statistic if np.std(pred_cc) > 0 else 0.0)
            ris[nome]["distinti"].append(len(np.unique(np.round(pred_cc, 10))) / len(pred_cc))
            nt, nc = T.shape[0], C.shape[0]
            pt = d.predict(np.repeat(T, nc, axis=0), np.tile(C, (nt, 1))).reshape(nt, nc)
            top = np.argpartition(pt, K - 1, axis=1)[:, :K]
            ris[nome]["overlap"].extend(len(set(a) & set(b)) / K for a, b in zip(top, top_vero))

print(f"prime 60 query di MQ2007, phi=1, foglie con piu di {K} documenti di training: {n_celle}")
for nome in modelli:
    r = ris[nome]
    print(f"{nome:9s} | Spearman con ED2 sulle coppie: mediana {np.median(r['rho']):.3f} | "
          f"vicini in comune con l'euclidea (top-5, test): media {np.mean(r['overlap']):.2f} | "
          f"quota valori distinti: mediana {np.median(r['distinti']):.2f} | fit {r['t']:.1f}s")
