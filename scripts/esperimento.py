"""
Esperimento unico per le fasi della roadmap.

Allena una variante su un dataset per ogni valore di |phi| richiesto, con i gruppi
di query formati per similarità come nel paper, e salva tutto quello che serve
per le analisi successive senza dover rifare il run:

  results/<dataset>/<variante>/phi<n>/seed<s>/
      config.json          parametri, gruppi di query, versione del codice
      metriche.json        NDCG@10 (media, deviazione standard, mediana) e tempi
      ndcg_per_query.csv   NDCG@10 di ogni query di test
      predizioni.npz       query, etichette e predizioni sul test
      modello.pkl          il QueryRanker allenato

Per le varianti RTR si salvano anche le predizioni del solo primo stadio e del
kNN con distanza euclidea, che si ottengono dallo stesso modello allenato, e la
quota di documenti di test su cui la distanza appresa può cambiare lo score.
Per RTRwRuleCard si salvano anche le statistiche della GAM.

Un run già concluso (metriche.json presente) viene saltato, così un esperimento
lungo interrotto riparte da dove era arrivato.

Esempio:
  python scripts/esperimento.py --dataset FINDHR --variante rtrwrulecard --phi 1 10 --max-gruppi 5
"""
import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from joblib.externals import cloudpickle

from ltr_utility import ModelParam
from ltr_utility.dataset import load_by_query_dataset, load_query_similarity, DatasetName
from ltr_utility.model_selection.evaluation import evaluate
from ruletreerank import QueryRanker
from experiments.wrappers import (WrapperMixRTR, WrapperMixRTRRuleCard, WrapperKNN, WrapperLGBMRanker,
                                  RandomRanker)

K_NDCG = 10


def leggi_argomenti():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["FINDHR", "FINDHRLIST", "MQ"])
    p.add_argument("--variante", required=True, choices=["rtr", "rtrwrulecard", "knn", "lambdamart", "casuale"])
    p.add_argument("--phi", type=int, nargs="+", default=[1, 2, 4, 6, 10])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-gruppi", type=int, default=None,
                   help="usa solo i primi N gruppi di query per ogni |phi|, per le prove veloci")
    p.add_argument("--dati", type=Path, default=REPO / "datasets")
    p.add_argument("--risultati", type=Path, default=REPO / "results")
    p.add_argument("--n-jobs-leaf", type=int, default=1)
    p.add_argument("--senza-modello", action="store_true", help="non salvare il modello allenato")
    return p.parse_args()


def parametri_variante(variante, n_jobs_leaf):
    """Iperparametri di ogni variante, tutti in un punto."""
    # Fissati dal gruppo nella griglia del 27 agosto. Profondità, k, uso della differenza
    # e target della distanza sono quelli raccomandati nel paper: restano provvisori finché
    # non arriva la model selection.
    rtr = dict(sdt_depth=5, pdt_depth=4, n_neighbors=5, feature_diff=True, dist_objective="dist",
               feature_concat=True, feature_sq_diff=False, subsample=1.0, sdt_max_leaf_nodes=None,
               min_samples_split=2, verbose=False, n_jobs_leaf=n_jobs_leaf)
    match variante:
        case "rtr":
            return WrapperMixRTR, rtr
        case "rtrwrulecard":
            # lr = 1 come indicato da Landi il 15 settembre. Con 20 round la GAM si fermava
            # al limite in circa 3 gruppi su 4: il massimo è alto perché a fermarla sia la patience.
            # La patience 3 è provvisoria.
            return WrapperMixRTRRuleCard, {**rtr, "rulecard_lr": 1.0, "rulecard_max_n_iter": 100,
                                           "rulecard_patience": 3}
        case "knn":
            # k provvisorio, lo stesso del secondo stadio di RTR
            return WrapperKNN, dict(n_neighbors=5)
        case "lambdamart":
            # parametri di default di LightGBM, provvisori
            return WrapperLGBMRanker, dict(verbose=-1)
        case "casuale":
            return RandomRanker, {}
    raise ValueError(variante)


def gruppi_di_query(dataset, phi, query_disponibili, max_gruppi):
    # FINDHRLIST usa gli stessi gruppi di FINDHR, come nei notebook del gruppo
    cartella = {"FINDHR": "FINDHR", "FINDHRLIST": "FINDHR", "MQ": "MQ2007"}[dataset]
    tutti = load_query_similarity(REPO / "experiments/query_based" / cartella / "results")
    # pandas legge le chiavi del json come interi
    gruppi = [[int(q) for q in g] for g in tutti[phi]]
    mancanti = set(sum(gruppi, [])) - set(int(q) for q in query_disponibili)
    assert not mancanti, f"query dei gruppi assenti dal dataset: {sorted(mancanti)[:10]}"
    return gruppi[:max_gruppi] if max_gruppi else gruppi


def quota_distanza(ranker, X, q, k):
    """Classifica ogni documento di test in base al suo gruppo foglia-query di training.

    La distanza può cambiare lo score solo se il gruppo ha più di k documenti con voti
    diversi. Dentro una foglia r(x) è costante, quindi residui diversi vogliono dire
    voti diversi.
    """
    conte = Counter()
    for modello, qs in ranker._models_to_qs.items():
        maschera = np.isin(q, qs)
        if not maschera.any():
            continue
        foglie = np.asarray(modello._shallow_dt.apply(X[maschera]))
        for foglia, qq in zip(foglie.tolist(), q[maschera].tolist()):
            agg = modello._leaf_dist_map.get((foglia, int(qq)))
            if agg is None:
                conte["vuoto"] += 1
            elif agg._fit_X.shape[0] <= k:
                conte["al_piu_k"] += 1
            elif np.unique(np.round(np.asarray(agg._y).ravel(), 10)).size == 1:
                conte["oltre_k_stesso_voto"] += 1
            else:
                conte["oltre_k_voti_diversi"] += 1
    totale = sum(conte.values())
    return {c: conte[c] / totale for c in ["oltre_k_voti_diversi", "oltre_k_stesso_voto", "al_piu_k", "vuoto"]}


def statistiche_gam(ranker, max_round):
    motivi, rounds, tipi = Counter(), [], Counter()
    for modello in ranker._models_to_qs:
        for agg in modello._leaf_dist_map.values():
            d = agg.custom_metric_func
            motivi[str(d._fallback_reason)] += 1
            if d.gam_ is None:
                continue
            rounds.append(len(d.gam_.estimators_))
            nf = d.num_features_
            for colonne, _ in d.gam_.estimators_:
                # con feature_concat e feature_diff le colonne sono [a, b, |a - b|]
                c = colonne[0]
                tipi["a" if c < nf else "b" if c < 2 * nf else "differenza"] += 1
    rounds = np.asarray(rounds)
    termini = sum(tipi.values())
    return {
        "gruppi_foglia_query": sum(motivi.values()),
        "motivi_ripiego": dict(motivi),
        "round_medio": float(rounds.mean()) if rounds.size else None,
        "round_mediano": float(np.median(rounds)) if rounds.size else None,
        "quota_round_al_massimo": float(np.mean(rounds == max_round)) if rounds.size else None,
        "quota_termini_per_colonna": {t: n / termini for t, n in tipi.items()} if termini else {},
    }


def commit_corrente():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True)
        sporco = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True)
        return out.stdout.strip() + (" (con modifiche non committate)" if sporco.stdout.strip() else "")
    except OSError:
        return None


def esegui_phi(args, phi, train_valid, test, cls, params):
    cartella = args.risultati / args.dataset / args.variante / f"phi{phi}" / f"seed{args.seed}"
    if (cartella / "metriche.json").exists():
        print(f"|phi|={phi}: già fatto, salto ({cartella})", flush=True)
        return
    cartella.mkdir(parents=True, exist_ok=True)

    gruppi = gruppi_di_query(args.dataset, phi, train_valid.unique_q, args.max_gruppi)
    query = sorted(set(sum(gruppi, [])))
    tr, te = train_valid[query], test[query]
    X, q, y = np.asarray(te.x), np.asarray(te.q), np.asarray(te.y)

    config = {
        "dataset": args.dataset, "variante": args.variante, "phi": phi, "seed": args.seed,
        "max_gruppi": args.max_gruppi, "modello": cls.__name__, "parametri": params,
        "n_query": len(query), "documenti_train": int(len(tr.y)), "documenti_test": int(len(y)),
        "gruppi_di_query": gruppi, "commit": commit_corrente(), "data": datetime.now().isoformat(timespec="seconds"),
        "versioni": {p: version(p) for p in ["scikit-learn", "RuleTree", "numpy", "lightgbm"]},
    }
    (cartella / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # il seed controlla le parti casuali dei modelli; lo split dei dati è fisso nel loader
    np.random.seed(args.seed)
    t0 = time.time()
    ranker = QueryRanker(ranker=ModelParam(model=cls, param=params), q_per_model=phi, batch_query=gruppi)
    ranker.fit(train=tr)
    tempo_fit = time.time() - t0

    t0 = time.time()
    predizioni = {"completo": ranker.predict(X=X, q=q)}
    tempo_predict = time.time() - t0
    if args.variante in ("rtr", "rtrwrulecard"):
        predizioni["solo_r"] = ranker.predict(X=X, q=q, output="score")
        predizioni["knn_euclideo"] = ranker.predict(X=X, q=q, output="euclidian")

    inizi = np.r_[0, np.cumsum(te.group_count)[:-1]]
    per_query = pd.DataFrame({"query": q[inizi]})
    metriche = {"tempo_fit_s": round(tempo_fit, 2), "tempo_predict_s": round(tempo_predict, 2)}
    for nome, pred in predizioni.items():
        media, dev, mediana = evaluate(pred=pred, labels=y, groups_count=te.group_count, k=K_NDCG)
        metriche[f"ndcg@{K_NDCG}_{nome}"] = {"media": float(media), "std": float(dev), "mediana": float(mediana)}
        per_query[nome] = evaluate(pred=pred, labels=y, groups_count=te.group_count, k=K_NDCG, aggregated=False)
    if args.variante in ("rtr", "rtrwrulecard"):
        metriche["quota_documenti_test"] = quota_distanza(ranker, X, q, params["n_neighbors"])
    if args.variante == "rtrwrulecard":
        metriche["gam"] = statistiche_gam(ranker, params["rulecard_max_n_iter"])

    per_query.to_csv(cartella / "ndcg_per_query.csv", index=False)
    np.savez_compressed(cartella / "predizioni.npz", query=q, etichette=y, **predizioni)
    (cartella / "metriche.json").write_text(json.dumps(metriche, indent=2), encoding="utf-8")

    # il modello si salva per ultimo: se il salvataggio fallisce, metriche e predizioni restano
    if not args.senza_modello:
        try:
            with open(cartella / "modello.pkl", "wb") as f:
                cloudpickle.dump(ranker, f)
        except Exception as exc:
            print(f"|phi|={phi}: modello non salvato ({exc!r})", flush=True)

    completo = metriche[f"ndcg@{K_NDCG}_completo"]
    print(f"|phi|={phi}: {len(query)} query, NDCG@{K_NDCG} {completo['media']:.4f} "
          f"(std {completo['std']:.3f}), fit {tempo_fit:.1f}s -> {cartella}", flush=True)


def main():
    args = leggi_argomenti()
    cls, params = parametri_variante(args.variante, args.n_jobs_leaf)
    _, _, test, train_valid = load_by_query_dataset(args.dati, DatasetName[args.dataset], hold_out=(0.5, 0.2, 0.3),
                                                    verbose=False)
    print(f"{args.dataset}: {train_valid} | {test}", flush=True)
    for phi in args.phi:
        esegui_phi(args, phi, train_valid, test, cls, params)


if __name__ == "__main__":
    main()
