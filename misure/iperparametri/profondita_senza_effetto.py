"""Perche' sul braccio additivo la profondita' dei learner non cambia niente.

Nella selezione a dieci fold il braccio additivo sceglie `pdt_depth` 2 in tutti e
202 i gruppi dei due dataset. Non e' una preferenza: i tre valori della griglia
danno punteggi di convalida identici, quindi la griglia effettiva e' di 48
configurazioni e non 144, e il 2 e' solo il primo valore provato. Sul braccio con
il PDT lo stesso parametro cambia il punteggio in tutti e 202 i gruppi, quindi non
e' un difetto della misura.

Lo script fa i tre controlli nell'ordine in cui li ho fatti, e il secondo serve
perche' la prima spiegazione che avevo scritto era sbagliata:

1. i punteggi di convalida sono identici alle tre profondita'? (sui file di
   `model_selection_fold10/`, con il braccio con il PDT come controllo);
2. e' l'albero che si ferma da se'? No: su dati continui sintetici le tre
   profondita' danno predizioni diverse a ogni dimensione provata;
3. allora sta nei dati. Si contano i valori distinti di ogni feature e si allena
   un albero univariato colonna per colonna sulle coppie di una cella. Il terzo
   controllo va ripetuto su piu' estrazioni: su una sola cella il numero di
   colonne in cui la profondita' conta oscilla, perche' dipende da quanti valori
   della feature piu' sbilanciata finiscono nell'estrazione.

Va eseguito sul server, dove stanno i file della selezione e i dataset:
  ~/bin/micromamba run -n rtr python profondita_senza_effetto.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path("/home/vmanzo/RuleTreeRank-DS2026")
SELEZIONE = REPO / "model_selection_fold10"
DATI = REPO / "datasets"
sys.path.insert(0, str(REPO))

from RuleTree import RuleTreeRegressor
from RuleTree.stumps.regression.DecisionTreeStumpRegressor import DecisionTreeStumpRegressor
from ltr_utility.dataset import load_by_query_dataset, DatasetName

PHI = [1, 2, 4, 6, 10]
PROFONDITA = (2, 4, 6)
ALTRI = ["feature_diff", "n_neighbors", "sdt_depth", "dist_objective"]
ESTRAZIONI = 20


def albero(profondita):
    return RuleTreeRegressor(
        max_depth=profondita, random_state=0,
        base_stumps=DecisionTreeStumpRegressor(max_depth=1, random_state=0))


def punteggi_identici():
    """1. La profondita' cambia il punteggio di convalida?"""
    print("=" * 72)
    print("1. punteggi di convalida alle tre profondita'")
    for variante in ("rtrwrulecard", "rtr"):
        for dataset in ("FINDHR", "FINDHRLIST"):
            gruppi = pareggi = 0
            scelte = Counter()
            for phi in PHI:
                f = SELEZIONE / ("%s_%s_phi%d.json" % (dataset, variante, phi))
                if not f.exists():
                    print("   manca", f.name)
                    continue
                for voce in json.loads(f.read_text()):
                    gruppi += 1
                    valori = voce["results"]
                    blocchi = {}
                    for c, s in zip(voce["configs"], valori):
                        chiave = tuple(c[k] for k in ALTRI)
                        blocchi.setdefault(chiave, {})[c["pdt_depth"]] = s
                    pareggi += all(len(set(b.values())) == 1 for b in blocchi.values())
                    scelte[voce["configs"][int(voce["idx_best"])]["pdt_depth"]] += 1
            print("   %-13s %-11s gruppi %3d | senza effetto in %3d | scelte %s"
                  % (variante, dataset, gruppi, pareggi, dict(sorted(scelte.items()))))


def non_e_l_albero():
    """2. L'albero si ferma da se' prima della profondita' consentita?"""
    print("=" * 72)
    print("2. controllo su dati continui sintetici")
    for n in (10, 50, 400, 2000):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(n, 1))
        y = x[:, 0] ** 2 + rng.normal(scale=.1, size=n)
        pred = {d: albero(d).fit(x, y).predict(x) for d in PROFONDITA}
        uguali = all(np.allclose(pred[2], pred[d]) for d in PROFONDITA)
        print("   righe %5d | valori distinti %s | identiche %s"
              % (n, [len(np.unique(np.round(pred[d], 12))) for d in PROFONDITA], uguali))


def colonne_in_cui_conta(doc):
    """Su quali colonne della rappresentazione di coppia la profondita' cambia la funzione."""
    ia, ib = np.triu_indices(len(doc), k=1)
    z = np.hstack([doc[ia], doc[ib], np.abs(doc[ia] - doc[ib])])
    bersaglio = ((doc[ia] - doc[ib]) ** 2).sum(axis=1)
    diverse = []
    for j in range(z.shape[1]):
        col = z[:, j:j + 1]
        pred = {d: albero(d).fit(col, bersaglio).predict(col) for d in PROFONDITA}
        if not all(np.allclose(pred[2], pred[d]) for d in PROFONDITA):
            diverse.append(j)
    return z.shape[1], diverse


def sta_nei_dati():
    """3. Quanti valori prendono le feature, e la profondita' sulle colonne vere."""
    print("=" * 72)
    print("3. i valori distinti delle feature")
    _, _, _, train = load_by_query_dataset(
        DATI, DatasetName["FINDHR"], hold_out=(0.5, 0.2, 0.3), verbose=False)
    x = train.x
    colonne = getattr(train, "_columns_name", None)
    nomi = list(colonne) if colonne is not None and len(colonne) == x.shape[1] \
        else ["f%d" % i for i in range(x.shape[1])]
    for i, n in enumerate(nomi):
        u = np.unique(x[:, i])
        print("   %-18s valori %-28s conteggi %s"
              % (n, [round(float(v), 3) for v in u], [int((x[:, i] == v).sum()) for v in u]))

    print()
    print("   la profondita' sulle 27 colonne, su %d estrazioni per dimensione di cella"
          % ESTRAZIONI)
    for n_doc in (10, 40, 100, 300):
        conteggi, quali = [], Counter()
        for seme in range(ESTRAZIONI):
            rng = np.random.default_rng(seme)
            doc = x[rng.choice(x.shape[0], n_doc, replace=False)]
            n_col, diverse = colonne_in_cui_conta(doc)
            conteggi.append(len(diverse))
            quali.update(diverse)
        print("   cella da %3d documenti | colonne in cui conta: min %d, mediana %.1f, max %d "
              "su %d | colonne coinvolte %s"
              % (n_doc, min(conteggi), float(np.median(conteggi)), max(conteggi), n_col,
                 dict(sorted(quali.items()))))


if __name__ == "__main__":
    punteggi_identici()
    non_e_l_albero()
    sta_nei_dati()
