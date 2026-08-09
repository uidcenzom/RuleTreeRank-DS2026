"""
Analisi del fallback su MQ2007.

Conta in quali celle (foglia, query) finiscono i documenti di test, in base al
numero di documenti di training presenti nella stessa cella. Serve a rispondere
alla domanda se il ripiego sulla distanza euclidea possa falsare i risultati.

Tre fasce:
  meno di 3   -> qui scatta il ripiego di RuleCard (una cella cosi' piccola non
                 permette di allenare un modello di distanza)
  al massimo 5 -> qui il kNN veloce ignora comunque la distanza, perche' con
                 n_neighbors=5 e non piu' di 5 vicini restituisce la media dei
                 residui senza usare la distanza appresa
  piu' di 5   -> qui la distanza appresa (RuleCard o PDT) viene davvero usata

Lo script non allena RuleCard: la ripartizione dipende solo dai dati e
dall'albero del primo stadio, quindi e' veloce.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from collections import Counter

from RuleTree import RuleTreeRegressor
from ltr_utility.dataset import load_by_query_dataset, DatasetName

SEED = 7
K = 5           # n_neighbors usato negli esperimenti
SDT_DEPTH = 5   # profondita' albero del primo stadio, come nel paper
PHIS = [1, 2, 4, 6, 10]

train, valid, test, train_valid = load_by_query_dataset(
    REPO / "datasets", DatasetName.MQ, hold_out=(0.5, 0.2, 0.3))
print(f"train_valid={train_valid} | test={test}", flush=True)

Xtr, qtr, ytr = np.asarray(train_valid.x), np.asarray(train_valid.q), np.asarray(train_valid.y)
Xte, qte = np.asarray(test.x), np.asarray(test.q)

# query nell'ordine del dataset, come nel raggruppamento usato negli esperimenti
uniq_q = list(dict.fromkeys(qtr.tolist()))


def gruppi(phi):
    for i in range(0, len(uniq_q), phi):
        yield set(uniq_q[i:i + phi])


print(f"\n{'phi':>4} | {'<3 (ripiego)':>13} | {'<=5 (dist. non usata)':>21} | {'>5 (dist. usata)':>16}")
print("-" * 64)

for phi in PHIS:
    conte = Counter()
    n_test = 0
    for grp in gruppi(phi):
        tr_m = np.isin(qtr, list(grp))
        te_m = np.isin(qte, list(grp))
        if not tr_m.any() or not te_m.any():
            continue
        tree = RuleTreeRegressor(max_depth=SDT_DEPTH, min_samples_split=2, random_state=SEED)
        tree.fit(Xtr[tr_m], ytr[tr_m])
        foglia_tr = np.asarray(tree.apply(Xtr[tr_m]))
        foglia_te = np.asarray(tree.apply(Xte[te_m]))
        q_tr, q_te = qtr[tr_m], qte[te_m]

        # numero di documenti di training in ciascuna cella (foglia, query)
        dim_cella = Counter(zip(foglia_tr.tolist(), q_tr.tolist()))
        for lf, qq in zip(foglia_te.tolist(), q_te.tolist()):
            n = dim_cella.get((lf, qq), 0)
            n_test += 1
            if n < 3:
                conte["<3"] += 1
            elif n <= 5:
                conte["<=5"] += 1
            else:
                conte[">5"] += 1

    p3 = 100 * conte["<3"] / n_test
    p5 = 100 * (conte["<3"] + conte["<=5"]) / n_test
    pg = 100 * conte[">5"] / n_test
    print(f"{phi:>4} | {p3:>12.1f}% | {p5:>20.1f}% | {pg:>15.1f}%", flush=True)

print("\nLa colonna '<=5' e' cumulativa e include '<3'.")
print("La distanza appresa incide solo sulla fascia '>5'.")
