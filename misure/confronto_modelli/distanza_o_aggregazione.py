"""
Cambiare la distanza o sostituire la correzione dentro la cella: quale pesa di più.

Il secondo stadio di RTR fa due cose: impara una distanza e poi aggrega i residui
dei vicini con un kNN. Questo script confronta, a partire dallo stesso modello di
riferimento (RTR con il PDT), due interventi separati e misurati sugli stessi run:

  distanza      si sostituisce il PDT con la GAM di RuleCard, lasciando il kNN
  aggregazione  si sostituisce tutta la correzione dentro la cella con una foresta

Attenzione a come si legge il secondo intervento: la foresta non e' un aggregatore
che combina i vicini scelti dalla distanza. Non usa la distanza appresa e non cerca
nessun vicino, si addestra sui documenti della cella dalle feature al residuo. Quindi
i due interventi non sono simmetrici, il secondo cambia piu' cose del primo. La
colonna si chiama ancora `aggregazione` perche' e' il nome con cui le tabelle sono
gia' pubblicate.

Per ogni dataset e |phi| riporta la media sui seed, la differenza rispetto al
riferimento e un test di Wilcoxon appaiato sull'NDCG@10 delle singole query,
calcolato sui valori mediati fra i seed. È lo stesso trattamento usato per il
confronto fra PDT e GAM, così i due numeri sono confrontabili fra loro.

Uso:
  python distanza_o_aggregazione.py <cartella results> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PHI = [1, 2, 4, 6, 10]
DATASET = ["FINDHRLIST", "FINDHR"]

# nome della colonna -> cartella della variante
INTERVENTI = {
    "riferimento (PDT + kNN)": "rtr",
    "cambia la distanza (GAM + kNN)": "rtrwrulecard",
    "cambia l'aggregazione (PDT + foresta)": "rtr_forestainfoglia",
}


def run(cartelle, dataset, variante, phi):
    """Tutti i seed di un run: (ndcg complessivo, ndcg per query)."""
    visti, medie, per_query = set(), [], []
    for base in cartelle:
        for seed_dir in sorted(Path(base).glob(f"{dataset}/{variante}/phi{phi}/seed*")):
            if seed_dir.name in visti or not (seed_dir / "metriche.json").exists():
                continue
            visti.add(seed_dir.name)
            medie.append(json.loads((seed_dir / "metriche.json").read_text(encoding="utf-8"))
                         ["ndcg@10_completo"]["media"])
            f = seed_dir / "ndcg_per_query.csv"
            if f.exists():
                per_query.append(pd.read_csv(f).set_index("query")["completo"])
    return medie, per_query


def confronta(cartelle):
    righe = []
    for dataset in DATASET:
        for phi in PHI:
            dati = {nome: run(cartelle, dataset, variante, phi)
                    for nome, variante in INTERVENTI.items()}
            base_medie, base_query = dati["riferimento (PDT + kNN)"]
            if not base_medie:
                continue
            riga = {"dataset": dataset, "phi": phi, "seed": len(base_medie),
                    "riferimento": round(float(np.mean(base_medie)), 4)}
            base_q = pd.concat(base_query, axis=1).mean(axis=1) if base_query else None

            for nome in list(INTERVENTI)[1:]:
                medie, query = dati[nome]
                etichetta = "distanza" if "distanza" in nome else "aggregazione"
                if not medie:
                    riga[etichetta] = riga[f"guadagno_{etichetta}"] = None
                    continue
                riga[etichetta] = round(float(np.mean(medie)), 4)
                riga[f"guadagno_{etichetta}"] = round(float(np.mean(medie) - np.mean(base_medie)), 4)
                if base_q is not None and query:
                    q = pd.concat(query, axis=1).mean(axis=1)
                    d = (q - base_q).dropna()
                    # il gain e' la media delle differenze per query, calcolata PRIMA di
                    # scartare i pareggi: il Wilcoxon li toglie, la media no
                    riga[f"gain_{etichetta}"] = round(float(d.mean()), 4)
                    d = d[d != 0]
                    riga[f"p_{etichetta}"] = round(wilcoxon(d).pvalue, 4) if len(d) else None
            righe.append(riga)
    return pd.DataFrame(righe)


def main():
    cartelle = sys.argv[1:] or ["results_server", "results"]
    tabella = confronta(cartelle)
    if tabella.empty:
        print("nessun risultato trovato in", cartelle)
        return
    print(tabella.to_string(index=False))

    g_dist = tabella["guadagno_distanza"].dropna()
    g_aggr = tabella["guadagno_aggregazione"].dropna()
    print(f"\nguadagno medio cambiando la distanza:     {g_dist.mean():+.4f} "
          f"(da {g_dist.min():+.4f} a {g_dist.max():+.4f})")
    print(f"guadagno medio cambiando l'aggregazione: {g_aggr.mean():+.4f} "
          f"(da {g_aggr.min():+.4f} a {g_aggr.max():+.4f})")
    confronti = (tabella["guadagno_aggregazione"] > tabella["guadagno_distanza"]).sum()
    print(f"la correzione nella cella pesa più della distanza in {confronti} casi su {len(tabella)}")

    out = Path(__file__).with_suffix(".csv")
    tabella.to_csv(out, index=False)
    print("scritto", out)


if __name__ == "__main__":
    main()
