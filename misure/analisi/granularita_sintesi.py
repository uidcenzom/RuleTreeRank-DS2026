"""Riassume le misure di granularita' nei numeri citati dal capitolo 5.

Legge i CSV prodotti da granularita.py e stampa, per fascia di dimensione della cella,
quanti valori di distanza distinti producono i due modelli e quanto spesso il tetto
strutturale del PDT, cioe' il numero di foglie, viene effettivamente raggiunto.
"""
import csv
from pathlib import Path

import numpy as np

QUI = Path(__file__).parent
TETTO = 2 ** 4  # profondita' 4, quindi al piu' sedici foglie e sedici valori distinti
FASCE = ["6-10", "11-20", "21-50", "51+"]


def leggi(nome):
    with open(QUI / nome, encoding="utf-8") as f:
        righe = list(csv.DictReader(f))
    for r in righe:
        for k in ("documenti", "coppie", "distinti_pdt", "distinti_gam"):
            r[k] = int(r[k])
    return righe


for nome, etichetta in (("granularita_findhr.csv", "FINDHR"),
                        ("granularita_findhrlist.csv", "FINDHRLIST")):
    righe = leggi(nome)
    print("=" * 76)
    print(f"{etichetta}: {len(righe)} celle")
    print(f"{'fascia':>8} {'celle':>6} {'coppie':>8} {'PDT':>6} {'GAM':>6} {'GAM/PDT':>8}"
          f" {'GAM>PDT':>8} {'PDT al tetto':>13}")
    for f in FASCE:
        sel = [r for r in righe if r["fascia"] == f]
        if not sel:
            continue
        pdt = np.array([r["distinti_pdt"] for r in sel])
        gam = np.array([r["distinti_gam"] for r in sel])
        print(f"{f:>8} {len(sel):>6} {np.median([r['coppie'] for r in sel]):>8.0f}"
              f" {np.median(pdt):>6.0f} {np.median(gam):>6.0f}"
              f" {np.median(gam / pdt):>8.2f} {np.mean(gam > pdt):>8.0%}"
              f" {np.mean(pdt >= TETTO):>13.0%}")
    pdt = np.array([r["distinti_pdt"] for r in righe])
    gam = np.array([r["distinti_gam"] for r in righe])
    print(f"{'tutte':>8} {len(righe):>6} {np.median([r['coppie'] for r in righe]):>8.0f}"
          f" {np.median(pdt):>6.0f} {np.median(gam):>6.0f}"
          f" {np.median(gam / pdt):>8.2f} {np.mean(gam > pdt):>8.0%}"
          f" {np.mean(pdt >= TETTO):>13.0%}")
    # il confronto che conta per la tesi: il PDT non supera mai il proprio tetto,
    # la GAM si' e questo e' il vincolo che l'argomento strutturale prevede
    print(f"  massimo valori distinti: PDT {pdt.max()}, GAM {gam.max()}, tetto del PDT {TETTO}")
    print(f"  celle in cui la GAM supera il tetto del PDT: {np.mean(gam > TETTO):.0%}")
