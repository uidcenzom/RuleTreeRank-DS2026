"""Quanti valori di distanza distinti producono il PDT e la GAM, sulle celle vere.

La motivazione del capitolo 5 dice che il modello additivo distingue piu' finemente
dell'albero singolo. L'argomento strutturale e' solido: un albero con l foglie puo'
restituire al piu' l valori, una somma di learner no. Quello che non era misurato e'
se il vincolo sia davvero attivo sui nostri dati.

Questo script lo misura ricostruendo le celle come le costruisce la pipeline, cioe'
albero del primo stadio a profondita' 5 dentro una query, e addestrando i due modelli
di distanza con la configurazione dei run veri, non con una ridotta. La differenza
conta: lo script precedente usava learning_rate 0.2 e 20 iterazioni, mentre i run
usano learning_rate 1.0 e 100 iterazioni, quindi misurava una GAM molto piu' debole
di quella della tesi.

I risultati sono riportati per fascia di dimensione della cella, perche' su una cella
di n documenti le coppie sono n(n-1)/2: se le coppie sono meno delle foglie del PDT
il tetto dell'albero non e' raggiungibile e i due modelli non possono differire, quindi
una media su tutte le celle direbbe soprattutto quanto sono piccole le celle.

Uso:
  python granularita.py [--dataset FINDHR] [--per-fascia 12] [--seed 7]
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

def trova_repo():
    """Il repo sta sopra questo file, ma lo script vive in due alberi diversi."""
    for base in Path(__file__).resolve().parents:
        if (base / "ruletreerank").is_dir():
            return base
        if (base / "RuleTreeRank-DS2026" / "ruletreerank").is_dir():
            return base / "RuleTreeRank-DS2026"
    raise RuntimeError("repo RuleTreeRank-DS2026 non trovato sopra " + __file__)


REPO = trova_repo()
sys.path.insert(0, str(REPO))

from RuleTree import RuleTreeRegressor
from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from ltr_utility import ModelParam
from ltr_utility.dataset import load_by_query_dataset, DatasetName
from ruletreerank import PairwiseDistanceTree, RuleCardPairwiseDistance

# le fasce di dimensione della cella, scelte sui quantili osservati: la prima e' quella
# in cui il tetto del PDT non puo' essere attivo, le ultime quelle in cui lo e' di sicuro
FASCE = [(6, 10), (11, 20), (21, 50), (51, 10 ** 6)]


def nome_fascia(n):
    for lo, hi in FASCE:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
    return None


def modelli(profondita, seed):
    """Gli stessi parametri dei run: vedi experiments/varianti.py e scripts/esperimento.py."""
    rappresentazione = {
        "base_regressor": ModelParam(RuleTreeRegressor, {
            "max_depth": profondita, "random_state": seed,
            "base_stumps": DecisionTreeStumpRegressor(max_depth=1, random_state=seed)}),
        "feature_concat": True,
        "feature_diff": True,
        "feature_sq_diff": False,
        "subsample": 1.0,
        "verbose": False,
    }
    return {
        "pdt": lambda: PairwiseDistanceTree(**rappresentazione),
        "gam": lambda: RuleCardPairwiseDistance(**rappresentazione, learning_rate=1.0,
                                                max_n_iter=100, patience=15),
    }


def celle(train_valid, profondita_srt, seed):
    """Le celle della pipeline a |phi| = 1: una foglia dentro una query."""
    X, y, q = np.asarray(train_valid.x), np.asarray(train_valid.y), np.asarray(train_valid.q)
    trovate = []
    for query in np.unique(q):
        m = q == query
        # il seme va dato anche allo stump: RuleTreeRegressor non gli passa il proprio
        # random_state e senza questo le celle cambiano da un'esecuzione all'altra
        albero = RuleTreeRegressor(
            max_depth=profondita_srt, min_samples_split=2, random_state=seed,
            base_stumps=DecisionTreeStumpRegressor(max_depth=1, random_state=seed),
        ).fit(X[m], y[m])
        foglie = np.asarray(albero.apply(X[m]))
        Xq = X[m]
        for foglia in np.unique(foglie):
            trovate.append((int(query), str(foglia), Xq[foglie == foglia]))
    return trovate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="FINDHR")
    p.add_argument("--per-fascia", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--pdt-depth", type=int, default=4)
    p.add_argument("--srt-depth", type=int, default=5)
    p.add_argument("--uscita", default=None)
    a = p.parse_args()

    _, _, _, train_valid = load_by_query_dataset(REPO / "datasets", DatasetName[a.dataset],
                                                 hold_out=(0.5, 0.2, 0.3), verbose=False)
    tutte = celle(train_valid, a.srt_depth, a.seed)
    print(f"{a.dataset}: {len(tutte)} celle totali a |phi| = 1", flush=True)

    # distribuzione completa, calcolata su tutte le celle e non sul campione
    dimensioni = np.array([c.shape[0] for _, _, c in tutte])
    print(f"  dimensione: mediana {np.median(dimensioni):.0f}, "
          f"quota con al piu' 5 documenti {np.mean(dimensioni <= 5):.0%}, "
          f"massimo {dimensioni.max()}", flush=True)

    # campione stratificato: le celle grandi sono poche e sono quelle che discriminano,
    # quindi vanno pescate apposta invece di lasciarle al caso
    rng = np.random.default_rng(a.seed)
    per_fascia = defaultdict(list)
    for cella in tutte:
        f = nome_fascia(cella[2].shape[0])
        if f:
            per_fascia[f].append(cella)
    campione = []
    for f, elenco in per_fascia.items():
        idx = rng.choice(len(elenco), min(a.per_fascia, len(elenco)), replace=False)
        campione.extend(elenco[i] for i in idx)
    print(f"  campione: {len(campione)} celle, "
          + ", ".join(f"{f} n={min(a.per_fascia, len(e))} su {len(e)}"
                      for f, e in sorted(per_fascia.items())), flush=True)

    crea = modelli(a.pdt_depth, a.seed)
    righe = []
    t0 = time.time()
    for i, (query, foglia, C) in enumerate(campione, 1):
        n = C.shape[0]
        ia, ib = np.triu_indices(n, 1)
        riga = {"dataset": a.dataset, "query": query, "foglia": foglia,
                "documenti": n, "coppie": len(ia), "fascia": nome_fascia(n)}
        for nome, costruisci in crea.items():
            d = costruisci()
            t = time.time()
            d.fit(C)
            pred = d.predict(C[ia], C[ib])
            # arrotondamento a 10 decimali: due valori che differiscono oltre quella cifra
            # sono rumore di virgola mobile, non una distinzione che il modello fa davvero
            distinti = len(np.unique(np.round(pred, 10)))
            riga[f"distinti_{nome}"] = distinti
            riga[f"quota_{nome}"] = distinti / len(ia)
            riga[f"tempo_{nome}"] = round(time.time() - t, 2)
        righe.append(riga)
        if i % 5 == 0 or i == len(campione):
            print(f"  {i}/{len(campione)} celle, {time.time() - t0:.0f}s", flush=True)

    uscita = Path(a.uscita) if a.uscita else Path(__file__).with_name(
        f"granularita_{a.dataset.lower()}.csv")
    with open(uscita, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)

    print()
    print(f"{'fascia':>8} {'celle':>6} {'coppie':>8} {'distinti PDT':>13} {'distinti GAM':>13}"
          f" {'quota PDT':>10} {'quota GAM':>10}")
    for lo, hi in FASCE:
        f = f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
        sel = [r for r in righe if r["fascia"] == f]
        if not sel:
            continue
        print(f"{f:>8} {len(sel):>6} {np.median([r['coppie'] for r in sel]):>8.0f}"
              f" {np.median([r['distinti_pdt'] for r in sel]):>13.0f}"
              f" {np.median([r['distinti_gam'] for r in sel]):>13.0f}"
              f" {np.median([r['quota_pdt'] for r in sel]):>10.2f}"
              f" {np.median([r['quota_gam'] for r in sel]):>10.2f}")
    print()
    print(f"scritto {uscita}")


if __name__ == "__main__":
    main()
