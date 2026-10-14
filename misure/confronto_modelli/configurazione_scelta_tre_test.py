"""Il confronto a configurazione scelta con i tre test appaiati, non solo Wilcoxon.

`configurazione_scelta_fold10.py` dice che a configurazione scelta il guadagno e'
+0.0007 e non e' significativo in nessuna combinazione, ma usa solo Wilcoxon.
Smucker, Allan e Carterette (CIKM 2007) trovano che Wilcoxon rileva la
significativita' peggio del t appaiato e del test di permutazione. A
configurazione fissa questo va a nostro sfavore, quindi non preoccupa: Wilcoxon
dichiara significative sei combinazioni contro le otto degli altri due. Qui
invece la conclusione e' "nessuna differenza", e un test meno potente potrebbe
non vederne una che c'e'. Questo script rifa' il confronto con t e permutazione.

Il test di permutazione scambia a caso i due bracci su ogni query, cioe' cambia
segno alla differenza per query, con 10^5 ricampionamenti come nel capitolo 6.

Uso:
  python configurazione_scelta_tre_test.py <cartella dei risultati>
"""
import sys

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

from configurazione_scelta_fold10 import DATASET, PHI, leggi

RICAMPIONAMENTI = 100_000


def permutazione(d, generatore):
    osservato = abs(d.mean())
    segni = generatore.choice([-1.0, 1.0], size=(RICAMPIONAMENTI, len(d)))
    nulle = np.abs((segni * d).mean(axis=1))
    return (1 + np.sum(nulle >= osservato)) / (RICAMPIONAMENTI + 1)


def main():
    radice = sys.argv[1] if len(sys.argv) > 1 else "scratch_fold10"
    dati = leggi(radice)
    generatore = np.random.default_rng(0)
    print(f"{'dataset':<11} {'phi':>4} {'guadagno':>9} {'Wilcoxon':>9} {'t':>8} {'perm':>8}")
    conta = {"Wilcoxon": 0, "t": 0, "perm": 0}
    for dataset in DATASET:
        for phi in PHI:
            semi = sorted(s for (d, p, s) in dati if d == dataset and p == phi)
            differenze = []
            for s in semi:
                v = dati[(dataset, phi, s)]
                if "pdt" in v and "gam" in v:
                    differenze.append(v["gam"]["completo"].values - v["pdt"]["completo"].values)
            if not differenze:
                continue
            d = np.mean(differenze, axis=0)
            p = {"Wilcoxon": wilcoxon(d).pvalue,
                 "t": ttest_1samp(d, 0.0).pvalue,
                 "perm": permutazione(d, generatore)}
            for k in conta:
                conta[k] += p[k] < 0.05
            print(f"{dataset:<11} {phi:>4} {d.mean():>+9.4f} {p['Wilcoxon']:>9.3f}"
                  f" {p['t']:>8.3f} {p['perm']:>8.3f}")
    print()
    print("combinazioni significative a 0.05:", conta)


if __name__ == "__main__":
    main()
