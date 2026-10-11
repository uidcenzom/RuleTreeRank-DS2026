"""Il confronto fra i due bracci a configurazione scelta per gruppo, dieci fold.

A configurazione fissa il modello additivo guadagna +0.0038, positivo in tutte e
dieci le combinazioni e significativo in sei. Resta da sapere se il vantaggio regge
quando ciascun braccio riceve la configurazione piu' adatta a se', cioe' quando il
confronto non e' piu' fra due modelli nella stessa impostazione ma fra due modelli al
loro meglio. E' il limite che il capitolo delle conclusioni dichiara apertamente.

La selezione a tre fold non bastava, perche' su gruppi piccoli sceglieva in larga
parte rumore. Questa usa dieci fold su entrambi i bracci.

Il confronto e' appaiato sulle query e mediato sui semi, come nel resto del capitolo
sperimentale, cosi' i numeri sono confrontabili con quelli a configurazione fissa.

Uso:
  python configurazione_scelta_fold10.py <cartella dei risultati>
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PHI = [1, 2, 4, 6, 10]
DATASET = ["FINDHR", "FINDHRLIST"]
BRACCI = {"rtr_scelta": "pdt", "rtrwrulecard_scelta": "gam"}


def leggi(radice):
    """NDCG per query di ogni run, indicizzato per dataset, phi, seme e braccio."""
    fuori = defaultdict(dict)
    for f in Path(radice).rglob("ndcg_per_query.csv"):
        dataset, variante, phi, seme = f.parts[-5], f.parts[-4], f.parts[-3], f.parts[-2]
        braccio = BRACCI.get(variante)
        if braccio is None:
            continue
        df = pd.read_csv(f).sort_values("query")
        chiave = (dataset, int(phi.replace("phi", "")), int(seme.replace("seed", "")))
        fuori[chiave][braccio] = df
    return fuori


def main():
    radice = sys.argv[1] if len(sys.argv) > 1 else "scratch_fold10"
    dati = leggi(radice)
    print("run letti:", sum(len(v) for v in dati.values()))

    righe = []
    for dataset in DATASET:
        for phi in PHI:
            semi = sorted(s for (d, p, s) in dati if d == dataset and p == phi)
            differenze, pdt_m, gam_m = [], [], []
            for s in semi:
                v = dati[(dataset, phi, s)]
                if "pdt" not in v or "gam" not in v:
                    continue
                a, b = v["pdt"], v["gam"]
                assert (a["query"].values == b["query"].values).all(), "query disallineate"
                differenze.append(b["completo"].values - a["completo"].values)
                pdt_m.append(a["completo"].mean())
                gam_m.append(b["completo"].mean())
            if not differenze:
                continue
            # media sui semi della differenza per query, poi test appaiato sulle query
            media_per_query = np.mean(differenze, axis=0)
            p = wilcoxon(media_per_query).pvalue if np.any(media_per_query) else 1.0
            righe.append({
                "dataset": dataset, "phi": phi, "semi": len(differenze),
                "pdt": round(float(np.mean(pdt_m)), 4),
                "gam": round(float(np.mean(gam_m)), 4),
                "guadagno": round(float(media_per_query.mean()), 4),
                "p": round(float(p), 4),
            })

    uscita = Path(__file__).with_name("configurazione_scelta_fold10.csv")
    with open(uscita, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)

    print()
    print(f"{'dataset':<11} {'phi':>4} {'PDT':>8} {'GAM':>8} {'guadagno':>10} {'p':>8}")
    for r in righe:
        print(f"{r['dataset']:<11} {r['phi']:>4} {r['pdt']:>8.4f} {r['gam']:>8.4f}"
              f" {r['guadagno']:>+10.4f} {r['p']:>8.3f}")
    g = [r["guadagno"] for r in righe]
    sig = sum(1 for r in righe if r["p"] < 0.05)
    print()
    print("guadagno medio %+.4f, positivo in %d su %d, significativo in %d su %d"
          % (float(np.mean(g)), sum(1 for x in g if x > 0), len(g), sig, len(righe)))
    print("per confronto, a configurazione fissa: +0.0038, 10 su 10, significativo in 6")
    print("rumore del seme: deviazione standard 0.0010, 95esimo percentile 0.0023")
    print("scritto", uscita)


if __name__ == "__main__":
    main()
