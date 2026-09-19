"""
I due modelli con la configurazione scelta per ogni gruppo di query.

Confronta quattro cose, per ogni dataset e ogni |phi|:

  rtr                  PDT, una configurazione fissa uguale per tutti i gruppi
  rtr_scelta           PDT, la configurazione scelta dalla model selection per ogni gruppo
  rtrwrulecard         GAM, configurazione fissa
  rtrwrulecard_scelta  GAM, configurazione scelta per ogni gruppo

Due avvertenze, che valgono anche per come si leggono i numeri.

La prima: il confronto fra configurazione fissa e configurazione scelta mescola due
cambiamenti, perche' i run a configurazione fissa usano i raggruppamenti di query del
repository e quelli a configurazione scelta quelli del gruppo di ricerca, e le due
partizioni coincidono solo per |phi| = 1. Quel confronto va quindi letto come "il
protocollo completo contro quello provvisorio", non come l'effetto della sola ricerca
degli iperparametri.

La seconda: il confronto fra i due modelli dentro lo stesso regime e' invece pulito,
perche' cambia solo il modello di distanza. E' quello che risponde alla domanda della
tesi.

Il test di Wilcoxon e' appaiato sull'NDCG@10 delle singole query, mediato fra i seed.

Uso:
  python configurazione_scelta.py <cartella scelta> [<cartella fissa> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PHI = [1, 2, 4, 6, 10]
DATASET = ["FINDHRLIST", "FINDHR"]
VARIANTI = {"PDT fissa": "rtr", "PDT scelta": "rtr_scelta",
            "GAM fissa": "rtrwrulecard", "GAM scelta": "rtrwrulecard_scelta"}


def run(cartelle, dataset, variante, phi):
    """Media sui seed e NDCG per query mediato sui seed, o None se il run non c'e'."""
    medie, per_query, visti = [], [], set()
    for base in cartelle:
        for d in sorted(Path(base).glob(f"{dataset}/{variante}/phi{phi}/seed*")):
            if d.name in visti or not (d / "metriche.json").exists():
                continue
            visti.add(d.name)
            medie.append(json.loads((d / "metriche.json").read_text(encoding="utf-8"))
                         ["ndcg@10_completo"]["media"])
            f = d / "ndcg_per_query.csv"
            if f.exists():
                per_query.append(pd.read_csv(f).set_index("query")["completo"])
    if not medie:
        return None
    return {"seed": len(medie), "media": float(np.mean(medie)),
            "std": float(np.std(medie, ddof=1)) if len(medie) > 1 else 0.0,
            "per_query": pd.concat(per_query, axis=1).mean(axis=1) if per_query else None}


def confronto(a, b):
    """Differenza fra due run e p del test appaiato sulle query, se possibile."""
    if a is None or b is None:
        return None, None
    differenza = b["media"] - a["media"]
    p = None
    if a["per_query"] is not None and b["per_query"] is not None:
        d = (b["per_query"] - a["per_query"]).dropna()
        d = d[d != 0]
        if len(d):
            p = float(wilcoxon(d).pvalue)
    return round(differenza, 4), (round(p, 4) if p is not None else None)


def main():
    cartelle = sys.argv[1:] or ["risultati_scelta", "results_server", "results"]
    righe = []
    for dataset in DATASET:
        for phi in PHI:
            dati = {nome: run(cartelle, dataset, variante, phi) for nome, variante in VARIANTI.items()}
            riga = {"dataset": dataset, "phi": phi}
            for nome, d in dati.items():
                riga[nome] = round(d["media"], 4) if d else None
                riga[f"seed {nome}"] = d["seed"] if d else 0
            # quanto rende scegliere la configurazione, per ciascun modello
            riga["PDT: scelta - fissa"], riga["p PDT"] = confronto(dati["PDT fissa"], dati["PDT scelta"])
            riga["GAM: scelta - fissa"], riga["p GAM"] = confronto(dati["GAM fissa"], dati["GAM scelta"])
            # il confronto fra i due modelli, dentro ciascun regime
            riga["GAM - PDT, fissa"], riga["p fissa"] = confronto(dati["PDT fissa"], dati["GAM fissa"])
            riga["GAM - PDT, scelta"], riga["p scelta"] = confronto(dati["PDT scelta"], dati["GAM scelta"])
            righe.append(riga)

    t = pd.DataFrame(righe)
    valori = ["dataset", "phi", "PDT fissa", "PDT scelta", "GAM fissa", "GAM scelta"]
    print("=== NDCG@10 medio sui seed ===")
    print(t[valori].to_string(index=False))

    print("\n=== quanto rende scegliere la configurazione per gruppo ===")
    print("(attenzione: cambia anche il raggruppamento delle query, tranne per |phi| = 1)")
    print(t[["dataset", "phi", "PDT: scelta - fissa", "p PDT", "GAM: scelta - fissa", "p GAM"]].to_string(index=False))

    print("\n=== il confronto fra i due modelli, a parita' di tutto il resto ===")
    print(t[["dataset", "phi", "GAM - PDT, fissa", "p fissa", "GAM - PDT, scelta", "p scelta"]].to_string(index=False))

    for etichetta in ["GAM - PDT, fissa", "GAM - PDT, scelta"]:
        d = t[etichetta].dropna()
        if len(d):
            print(f"\n{etichetta}: media {d.mean():+.4f}, positivo in {(d > 0).sum()} casi su {len(d)}")

    mancanti = [c for c in VARIANTI if t[f"seed {c}"].sum() == 0]
    if mancanti:
        print("\nrun non ancora disponibili:", ", ".join(mancanti))

    out = Path(__file__).with_suffix(".csv")
    t.to_csv(out, index=False)
    print("\nscritto", out)


if __name__ == "__main__":
    main()
