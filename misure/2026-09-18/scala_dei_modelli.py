"""
La scala dei modelli: dal ranking casuale ai competitor, su tutti i |phi|.

Mette in una tabella sola tutto quello che abbiamo già misurato, così si vede a
colpo d'occhio quanto vale ogni pezzo del modello:

  casuale           il minimo, nessuna informazione
  solo r(x)         solo il primo stadio, senza la correzione dei vicini
  euclidea          kNN dentro la foglia con la distanza euclidea, cioè RTR con
                    una distanza non appresa
  RTR               distanza appresa con il PDT
  RTRwRuleCard      distanza appresa con la GAM
  kNN               kNN sull'intero gruppo di query, senza foglie
  LambdaMART        il competitor forte, senza interpretabilità

e i tre upper bound della fase 2, che rinunciano all'interpretabilità in punti
diversi della pipeline:

  foresta in foglia   stesso primo stadio, ma dentro la cella una foresta al
                      posto del kNN con distanza appresa
  foresta sul gruppo  una foresta su tutti i documenti del gruppo di query,
                      senza foglie e senza distanza
  boosting sul gruppo lo stesso con un gradient boosting

Le prime tre colonne e le due nostre escono dallo stesso run: ogni esperimento
RTR salva anche le predizioni del solo primo stadio e quelle del kNN euclideo.

I risultati si leggono da più cartelle (i run sul server hanno cinque seed, quelli
sul pc uno solo) e per ogni casella si riportano media e deviazione standard fra i
seed disponibili.

Uso:
  python scala_dei_modelli.py [cartella ...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PHI = [1, 2, 4, 6, 10]
DATASET = ["FINDHRLIST", "FINDHR"]

# come si chiama ogni colonna e da quale run e quale predizione arriva
COLONNE = [
    ("casuale", "casuale", "completo"),
    ("solo r(x)", "rtr", "solo_r"),
    ("euclidea in foglia", "rtr", "knn_euclideo"),
    ("RTR (PDT)", "rtr", "completo"),
    ("RTRwRuleCard (GAM)", "rtrwrulecard", "completo"),
    ("kNN", "knn", "completo"),
    ("LambdaMART", "lambdamart", "completo"),
    ("foresta in foglia", "rtr_forestainfoglia", "completo"),
    ("foresta sul gruppo", "foresta", "completo"),
    ("boosting sul gruppo", "boosting", "completo"),
]


def raccogli(cartelle):
    """Tutti gli NDCG@10 trovati: (dataset, variante, phi, predizione) -> lista sui seed.

    Lo stesso seed può stare in più cartelle, per esempio i run del server scaricati
    sul pc accanto a quelli fatti sul pc: vale il primo della lista, così un seed non
    viene contato due volte nella media.
    """
    unici = {}
    for base in cartelle:
        for f in Path(base).glob("*/*/phi*/seed*/metriche.json"):
            dataset, variante, phi, seed = f.parts[-5], f.parts[-4], int(f.parts[-3][3:]), f.parts[-2]
            metriche = json.loads(f.read_text(encoding="utf-8"))
            for chiave, m in metriche.items():
                if chiave.startswith("ndcg@10_"):
                    unici.setdefault((dataset, variante, phi, chiave[len("ndcg@10_"):], seed), m["media"])

    valori = defaultdict(list)
    for (dataset, variante, phi, predizione, _), media in unici.items():
        valori[(dataset, variante, phi, predizione)].append(media)
    return valori


def cella(valori, dataset, variante, phi, predizione):
    v = valori.get((dataset, variante, phi, predizione), [])
    if not v:
        return "", 0
    if len(v) == 1:
        return f"{v[0]:.4f}", 1
    return f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}", len(v)


def main():
    cartelle = sys.argv[1:] or ["results_server", "results"]
    valori = raccogli(cartelle)
    if not valori:
        print("nessun risultato trovato in", cartelle)
        return

    righe, seed_usati = [], defaultdict(set)
    for dataset in DATASET:
        for phi in PHI:
            riga = {"dataset": dataset, "phi": phi}
            for nome, variante, predizione in COLONNE:
                riga[nome], n = cella(valori, dataset, variante, phi, predizione)
                seed_usati[nome].add(n)
            righe.append(riga)

    tabella = pd.DataFrame(righe)
    print(tabella.to_string(index=False))
    print("\nseed per colonna:", {k: sorted(v) for k, v in seed_usati.items()})
    out = Path(__file__).with_suffix(".csv")
    tabella.to_csv(out, index=False)
    print("scritto", out)


if __name__ == "__main__":
    main()
