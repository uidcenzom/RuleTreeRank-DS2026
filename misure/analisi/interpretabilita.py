"""Quanto e' lunga e quanto e' stabile una spiegazione del modello additivo.

La tesi sostiene che le spiegazioni sono leggibili, ma fino a ora lo diceva senza
misurarlo, e lo dichiarava fra i limiti. Uno studio con utenti non e' alla nostra
portata; due misure surrogate pero' si fanno, e dicono qualcosa di concreto.

La prima e' la lunghezza. Una spiegazione del modello additivo e' una somma di
contributi con segno, uno per regola trattenuta dal boosting. Se le regole sono cento
e contano tutte uguale, "leggibile" e' una parola vuota; se poche regole portano quasi
tutto il valore, la spiegazione vera e' corta anche quando il modello e' grande. Qui si
contano le regole che servono per arrivare al 90% del contributo assoluto.

La seconda e' la stabilita'. Se due semi diversi producono spiegazioni diverse per la
stessa coppia, la spiegazione e' un artefatto del caso e non una proprieta' dei dati.
Si confrontano le feature usate dai due modelli e l'ordinamento delle distanze che
producono.

Uso:
  python interpretabilita.py [--dataset FINDHR] [--per-fascia 15] [--seme 7] [--seme-b 8]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def trova_repo():
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
from ruletreerank import RuleCardPairwiseDistance

FASCE = [(6, 10), (11, 20), (21, 50), (51, 10 ** 6)]


def nome_fascia(n):
    for lo, hi in FASCE:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
    return None


def crea_gam(profondita, seme):
    """Gli stessi parametri dei run: vedi scripts/esperimento.py."""
    return RuleCardPairwiseDistance(
        base_regressor=ModelParam(RuleTreeRegressor, {
            "max_depth": profondita, "random_state": seme,
            "base_stumps": DecisionTreeStumpRegressor(max_depth=1, random_state=seme)}),
        feature_concat=True, feature_diff=True, feature_sq_diff=False,
        subsample=1.0, verbose=False,
        learning_rate=1.0, max_n_iter=100, patience=15,
    )


def celle(train_valid, profondita_srt, seme):
    """Le celle della pipeline a |phi| = 1: una foglia dentro una query."""
    X, y, q = np.asarray(train_valid.x), np.asarray(train_valid.y), np.asarray(train_valid.q)
    trovate = []
    for query in np.unique(q):
        m = q == query
        albero = RuleTreeRegressor(
            max_depth=profondita_srt, min_samples_split=2, random_state=seme,
            base_stumps=DecisionTreeStumpRegressor(max_depth=1, random_state=seme),
        ).fit(X[m], y[m])
        foglie = np.asarray(albero.apply(X[m]))
        Xq = X[m]
        for foglia in np.unique(foglie):
            trovate.append((int(query), str(foglia), Xq[foglie == foglia]))
    return trovate


def contributi(modello, x_a, x_b):
    """Il contributo di ogni alberello alla distanza di una coppia.

    Ricalca `_raw_predict`: la previsione e' la somma dei contributi piu' una base,
    quindi i contributi sono esattamente gli addendi della spiegazione.
    """
    g = modello.gam_
    x_a, x_b = np.asarray(x_a, dtype=float), np.asarray(x_b, dtype=float)
    X = np.hstack([x_a, x_b])
    if g.use_difference:
        diff = np.abs(x_a - x_b)
        X = np.hstack([X, diff]) if g.use_pairwise else diff
    out = np.zeros((X.shape[0], len(g.estimators_)))
    feature_usate = []
    for t, (feat_idx, est) in enumerate(g.estimators_):
        out[:, t] = g.learning_rate * est.predict(X[:, feat_idx].reshape(X.shape[0], -1))
        feature_usate.append(tuple(sorted(int(i) for i in feat_idx)))
    return out, feature_usate


def regole_per_il_novanta(c):
    """Quante regole servono per coprire il 90% del contributo assoluto di una coppia."""
    a = np.abs(c)
    totale = a.sum()
    if totale <= 0:
        return 0
    ordinati = np.sort(a)[::-1]
    return int(np.searchsorted(np.cumsum(ordinati), 0.9 * totale) + 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="FINDHR")
    p.add_argument("--per-fascia", type=int, default=15)
    p.add_argument("--seme", type=int, default=7)
    p.add_argument("--seme-b", type=int, default=8, help="secondo seme, per la stabilità")
    p.add_argument("--pdt-depth", type=int, default=4)
    p.add_argument("--srt-depth", type=int, default=5)
    a = p.parse_args()

    _, _, _, train_valid = load_by_query_dataset(REPO / "datasets", DatasetName[a.dataset],
                                                 hold_out=(0.5, 0.2, 0.3), verbose=False)
    tutte = celle(train_valid, a.srt_depth, a.seme)
    print(f"{a.dataset}: {len(tutte)} celle a |phi| = 1", flush=True)

    rng = np.random.default_rng(a.seme)
    per_fascia = defaultdict(list)
    for cella in tutte:
        f = nome_fascia(cella[2].shape[0])
        if f:
            per_fascia[f].append(cella)
    campione = []
    for f, elenco in per_fascia.items():
        idx = rng.choice(len(elenco), min(a.per_fascia, len(elenco)), replace=False)
        campione.extend(elenco[i] for i in idx)
    print(f"  campione: {len(campione)} celle", flush=True)

    righe = []
    for i, (query, foglia, C) in enumerate(campione, 1):
        n = C.shape[0]
        ia, ib = np.triu_indices(n, 1)

        ma = crea_gam(a.pdt_depth, a.seme)
        ma.fit(C)
        if ma.gam_ is None or not getattr(ma.gam_, "estimators_", None):
            continue  # cella che ripiega sull'euclidea: non ha spiegazione additiva
        ca, feat_a = contributi(ma, C[ia], C[ib])

        mb = crea_gam(a.pdt_depth, a.seme_b)
        mb.fit(C)
        stabile_feature, stabile_ordine = float("nan"), float("nan")
        if mb.gam_ is not None and getattr(mb.gam_, "estimators_", None):
            cb, feat_b = contributi(mb, C[ia], C[ib])
            sa, sb = set(feat_a), set(feat_b)
            stabile_feature = len(sa & sb) / len(sa | sb) if (sa | sb) else float("nan")
            pa, pb = ca.sum(axis=1), cb.sum(axis=1)
            if np.std(pa) > 0 and np.std(pb) > 0:
                stabile_ordine = spearmanr(pa, pb).statistic

        novanta = [regole_per_il_novanta(c) for c in ca]
        righe.append({
            "dataset": a.dataset, "query": query, "foglia": foglia,
            "documenti": n, "coppie": len(ia), "fascia": nome_fascia(n),
            "regole_totali": ca.shape[1],
            "regole_per_90pc_mediana": float(np.median(novanta)),
            "feature_distinte": len(set(feat_a)),
            "stabilita_feature": stabile_feature,
            "stabilita_ordine": stabile_ordine,
        })
        if i % 10 == 0 or i == len(campione):
            print(f"  {i}/{len(campione)} celle", flush=True)

    if not righe:
        raise SystemExit("nessuna cella ha prodotto un modello additivo")

    uscita = Path(__file__).with_name(f"interpretabilita_{a.dataset.lower()}.csv")
    with open(uscita, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)

    print()
    print(f"{'fascia':>8} {'celle':>6} {'regole':>8} {'per il 90%':>11} {'feature':>9}"
          f" {'stab. feat.':>12} {'stab. ordine':>13}")
    for lo, hi in FASCE:
        f = f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
        sel = [r for r in righe if r["fascia"] == f]
        if not sel:
            continue
        print(f"{f:>8} {len(sel):>6}"
              f" {np.median([r['regole_totali'] for r in sel]):>8.0f}"
              f" {np.median([r['regole_per_90pc_mediana'] for r in sel]):>11.0f}"
              f" {np.median([r['feature_distinte'] for r in sel]):>9.0f}"
              f" {np.nanmedian([r['stabilita_feature'] for r in sel]):>12.2f}"
              f" {np.nanmedian([r['stabilita_ordine'] for r in sel]):>13.2f}")
    print()
    print(f"  su tutte: {len(righe)} celle, regole mediane"
          f" {np.median([r['regole_totali'] for r in righe]):.0f},"
          f" per il 90% ne bastano {np.median([r['regole_per_90pc_mediana'] for r in righe]):.0f}")
    print("scritto", uscita)


if __name__ == "__main__":
    main()
