"""Quanto rende la selezione per gruppo, su ciascuno dei due bracci.

Il confronto fra i due bracci a configurazione scelta dice che sono
indistinguibili (+0.0007, nessuna combinazione significativa), ma non dice di chi
sia il merito del cambiamento rispetto alla configurazione fissa. Questo lo dice:
mette i due regimi uno accanto all'altro braccio per braccio, e il risultato e'
che la selezione rende molto piu' al braccio con il PDT che a quello additivo,
cioe' che la configurazione di riferimento era piu' adatta al secondo. E' il
numero che sta nella sezione sugli iperparametri della tesi.

Uso:
  python effetto_della_selezione.py
"""
import csv
from pathlib import Path

import numpy as np

CONFRONTO = Path(__file__).resolve().parents[1] / "confronto_modelli"
FISSA = CONFRONTO / "confronto_multiseed_server.csv"
SCELTA = CONFRONTO / "configurazione_scelta_fold10.csv"

PHI = [1, 2, 4, 6, 10]
DATASET = ["FINDHR", "FINDHRLIST"]


def leggi(percorso):
    with open(percorso, encoding="utf-8") as f:
        return {(r["dataset"], int(r["phi"])): r for r in csv.DictReader(f)}


def main():
    fissa, scelta = leggi(FISSA), leggi(SCELTA)
    for dataset in DATASET:
        print("=" * 72)
        print(dataset)
        print("%4s %10s %11s %8s %10s %11s %8s"
              % ("phi", "PDT fissa", "PDT scelta", "delta", "GAM fissa", "GAM scelta", "delta"))
        dp, dg = [], []
        for phi in PHI:
            f, s = fissa.get((dataset, phi)), scelta.get((dataset, phi))
            if not f or not s:
                continue
            pf, gf = float(f["pdt_media"]), float(f["gam_media"])
            ps, gs = float(s["pdt"]), float(s["gam"])
            dp.append(ps - pf)
            dg.append(gs - gf)
            print("%4d %10.4f %11.4f %+8.4f %10.4f %11.4f %+8.4f"
                  % (phi, pf, ps, ps - pf, gf, gs, gs - gf))
        print("     effetto medio della selezione: PDT %+.4f, GAM %+.4f"
              % (float(np.mean(dp)), float(np.mean(dg))))


if __name__ == "__main__":
    main()
