import sys, time
REPO = r"C:\Users\manzo\Desktop\fork\RuleTreeRank-DS2026"
sys.path.insert(0, REPO)
from pathlib import Path
from collections import Counter
import numpy as np
from ltr_utility.dataset import load_by_query_dataset, DatasetName
from experiments.wrappers import WrapperMixRTRRuleCard

np.random.seed(7)
tr, va, te, tv = load_by_query_dataset(Path(REPO) / "datasets", DatasetName.MQ, hold_out=(0.5, 0.2, 0.3))
P = dict(pdt_depth=4, feature_concat=True, feature_diff=True, feature_sq_diff=False, subsample=1.0, verbose=False,
         rulecard_lr=0.2, rulecard_max_n_iter=20, rulecard_patience=3, n_neighbors=5, sdt_depth=5,
         sdt_max_leaf_nodes=None, min_samples_split=2, dist_objective="dist", n_jobs_leaf=1)
NF = 46


def analizza(etichetta, gruppi):
    righe, t_tot = [], 0.0
    for g in gruppi:
        sub = tv[g]
        m = WrapperMixRTRRuleCard(**P)
        t0 = time.time()
        m.fit(np.asarray(sub.x), np.asarray(sub.y), np.asarray(sub.q))
        t_tot += time.time() - t0
        for (L, q), agg in m._leaf_dist_map.items():
            d = agg.custom_metric_func
            n = agg._fit_X.shape[0]
            r = {"n": n, "motivo": d._fallback_reason, "T": 0, "tipi": Counter(), "asim": np.nan}
            if d.gam_ is not None:
                est = d.gam_.estimators_
                r["T"] = len(est)
                for idx, _ in est:
                    c = idx[0]
                    r["tipi"]["a_j" if c < NF else ("b_j" if c < 2 * NF else "|a_j-b_j|")] += 1
                X = agg._fit_X
                rng = np.random.default_rng(0)
                ia, ib = rng.integers(0, n, 400), rng.integers(0, n, 400)
                keep = ia != ib
                dab, dba = d.predict(X[ia[keep]], X[ib[keep]]), d.predict(X[ib[keep]], X[ia[keep]])
                r["asim"] = float(np.mean(np.abs(dab - dba)) / max(np.mean(dab), 1e-12))
            righe.append(r)
    print(f"\n===== {etichetta} | modelli {len(gruppi)} | tempo fit totale {t_tot:.1f}s =====")
    print("gruppetti (foglia, query):", len(righe))
    print("motivi di ripiego:", dict(Counter(r["motivo"] for r in righe)))
    con_gam = [r for r in righe if r["T"] > 0]
    T = np.array([r["T"] for r in con_gam])
    print(f"gruppetti con GAM: {len(con_gam)} | n documenti mediana {np.median([r['n'] for r in con_gam]):.0f}")
    print(f"round T: media {T.mean():.1f} mediana {np.median(T):.0f} min {T.min()} max {T.max()} | T == 20: {100*np.mean(T == 20):.1f}%")
    for lo, hi in [(3, 5), (6, 10), (11, 20), (21, 1000)]:
        s = [r["T"] for r in con_gam if lo <= r["n"] <= hi]
        if s:
            print(f"   n in [{lo},{hi}]: gruppetti {len(s)}, T medio {np.mean(s):.1f}, T==20 {100*np.mean(np.array(s) == 20):.0f}%")
    tipi = sum((r["tipi"] for r in con_gam), Counter())
    tot = sum(tipi.values())
    print("termini per tipo di colonna:", {k: f"{100*v/tot:.1f}%" for k, v in tipi.items()})
    a = np.array([r["asim"] for r in con_gam if r["n"] > 5])
    print(f"asimmetria |d(a,b)-d(b,a)| / media d, gruppetti con n>5: mediana {np.median(a):.3f}, 90 perc {np.percentile(a, 90):.3f}")


uq = list(tv.unique_q)
analizza("phi=1, prime 30 query", [[q] for q in uq[:30]])
analizza("phi=10, prime 30 query", [uq[i:i + 10] for i in range(0, 30, 10)])
