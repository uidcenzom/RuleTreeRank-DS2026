"""Ogni affermazione di significativita' della tesi, rifatta con i tre test appaiati.

Il capitolo 6 usa Wilcoxon come test principale. Smucker, Allan e Carterette
(CIKM 2007) trovano che rileva la significativita' peggio del t appaiato e del
test di permutazione, quindi va controllato che nessuna conclusione dipenda da
quella scelta. Dove la tesi dice "significativo" un test piu' potente puo' solo
confermare; dove dice "non significativo" potrebbe smentire, ed e' li' che serve.

Confronti coperti, tutti sull'NDCG@10 per query mediato sui cinque semi:
  1. secondo stadio sul primo, braccio additivo contro PDT, a configurazione scelta
  2. selezione per gruppo a tre e a dieci fold contro la configurazione fissa (PDT)
  3. foresta nella cella contro RTR con il PDT
  4. le tre varianti interpretabili della tabella sulle componenti alternative
     (pesatura dei vicini, almeno 6 e almeno 10 documenti per foglia), il cui
     seme 0 sta in `risultati/` e i semi da 1 a 4 in `risultati_semi/`

Uso, dalla radice del repository:
  python misure/confronto_modelli/tre_test_sui_confronti.py [run a dieci fold]

I run riaddestrati con la selezione a dieci fold stanno in una cartella a parte,
`risultati_scelta_fold10/` per default, perche' hanno lo stesso nome di variante
di quelli a tre fold.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon

RISULTATI = Path("risultati")
SCELTA_TRE = Path("risultati_scelta")
SCELTA_DIECI = Path(sys.argv[1] if len(sys.argv) > 1 else "risultati_scelta_fold10")
SEMI = Path(sys.argv[2] if len(sys.argv) > 2 else "risultati_semi")
DATASET = ["FINDHR", "FINDHRLIST"]
PHI = [1, 2, 4, 6, 10]
RICAMPIONAMENTI = 100_000
GENERATORE = np.random.default_rng(0)


def per_query(radice, dataset, variante, phi, colonna="completo"):
    radici = radice if isinstance(radice, (list, tuple)) else [radice]
    file, visti = [], set()
    for r in radici:
        for f in sorted((r / dataset / variante / f"phi{phi}").glob("seed*/ndcg_per_query.csv")):
            if f.parent.name not in visti:
                visti.add(f.parent.name)
                file.append(f)
    serie = [pd.read_csv(f).set_index("query")[colonna] for f in file]
    if len(serie) < 5:
        return None
    return pd.concat(serie, axis=1).mean(axis=1)


def tre_p(d):
    d = np.asarray(d, dtype=float)
    segni = GENERATORE.choice([-1.0, 1.0], size=(RICAMPIONAMENTI, len(d)))
    nulle = np.abs((segni * d).mean(axis=1))
    perm = (1 + np.sum(nulle >= abs(d.mean()))) / (RICAMPIONAMENTI + 1)
    return wilcoxon(d[d != 0]).pvalue, ttest_1samp(d, 0.0).pvalue, perm


def stampa(titolo, righe):
    print(f"\n=== {titolo}")
    print(f"{'dataset':<11} {'phi':>4} {'diff':>8} {'Wilcoxon':>9} {'t':>7} {'perm':>7}")
    for dataset, phi, d in righe:
        w, t, p = tre_p(d)
        print(f"{dataset:<11} {phi:>4} {np.mean(d):>+8.4f} {w:>9.3f} {t:>7.3f} {p:>7.3f}")
    for nome, i in (("Wilcoxon", 0), ("t", 1), ("perm", 2)):
        ps = [(np.mean(d), tre_p(d)[i]) for _, _, d in righe]
        meglio = sum(1 for m, p in ps if p < 0.05 and m > 0)
        peggio = sum(1 for m, p in ps if p < 0.05 and m < 0)
        print(f"  {nome:<9} significativo e positivo {meglio}, significativo e negativo {peggio},"
              f" p minimo {min(p for _, p in ps):.3f}")


def main():
    righe = []
    for ds in DATASET:
        for phi in PHI:
            g = [per_query(SCELTA_DIECI, ds, "rtrwrulecard_scelta", phi, c) for c in ("completo", "solo_r")]
            t = [per_query(SCELTA_DIECI, ds, "rtr_scelta", phi, c) for c in ("completo", "solo_r")]
            if any(x is None for x in g + t):
                continue
            righe.append((ds, phi, ((g[0] - g[1]) - (t[0] - t[1])).dropna().values))
    stampa("1. secondo stadio, additivo meno PDT, configurazione scelta (tesi: mai significativo, p min 0.124)", righe)

    for nome, radice in (("tre fold", SCELTA_TRE), ("dieci fold", SCELTA_DIECI)):
        righe = []
        for ds in DATASET:
            for phi in PHI:
                fissa = per_query(RISULTATI, ds, "rtr", phi)
                scelta = per_query(radice, ds, "rtr_scelta", phi)
                if fissa is None or scelta is None:
                    continue
                righe.append((ds, phi, (scelta - fissa).dropna().values))
        stampa(f"2. selezione a {nome} meno configurazione fissa, PDT", righe)

    righe = []
    for ds in DATASET:
        for phi in PHI:
            base = per_query(RISULTATI, ds, "rtr", phi)
            foresta = per_query(RISULTATI, ds, "rtr_forestainfoglia", phi)
            if base is None or foresta is None:
                continue
            righe.append((ds, phi, (foresta - base).dropna().values))
    stampa("3. foresta nella cella meno RTR con il PDT", righe)

    for variante, nome in (("rtr_pesato", "pesatura dei vicini (tesi: peggiore 10, significativa 8)"),
                           ("rtr_minfoglia6", "almeno 6 per foglia (tesi: positiva 4, significativa 2)"),
                           ("rtr_minfoglia10", "almeno 10 per foglia (tesi: positiva 2, significativa 2)")):
        righe = []
        for ds in DATASET:
            for phi in PHI:
                base = per_query(RISULTATI, ds, "rtr", phi)
                v = per_query([RISULTATI, SEMI], ds, variante, phi)
                if base is None or v is None:
                    continue
                righe.append((ds, phi, (v - base).dropna().values))
        stampa(f"4. {nome}", righe)


if __name__ == "__main__":
    main()
