"""Quanto si puo' ridurre il training del modello di distanza prima che il ranking cali.

E' la prima delle strade proposte da Andrea: il sospetto e' che le coppie da cui si
impara la distanza siano ridondanti, e che si possa pagare molto meno senza perdere
qualita'. Le coppie di una cella sono n(n-1)/2, quindi dimezzare i documenti ne lascia
un quarto.

Legge i `metriche.json` dei run dello sweep, uno per combinazione di dataset, braccio,
frazione e seme, e produce la tabella e il grafico: frazione dei dati sulle ascisse,
NDCG@10 sulle ordinate, una curva per braccio.

Uso:
  python sottocampionamento.py <cartella dei risultati> [cartella delle figure ...]
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FRAZIONI = [0.1, 0.25, 0.5, 1.0]
BRACCI = {"rtr": "PDT", "rtrwrulecard": "GAM"}
DATASET = {"FINDHR": "FindHR", "FINDHRLIST": "FindHR$_\\ell$"}


def leggi(radice):
    """Un record per run: dataset, braccio, frazione, seme, NDCG e tempo di fit."""
    righe = []
    for f in Path(radice).rglob("metriche.json"):
        parti = f.parts
        dataset = parti[-4 - 1] if parti[-4] .startswith("phi") else None
        # .../<dataset>/<variante>/phi1/seed<N>/metriche.json
        dataset, variante, _, seme = parti[-5], parti[-4], parti[-3], parti[-2]
        m = re.match(r"^(rtr|rtrwrulecard)(?:_sub(\d+))?$", variante)
        if not m:
            continue
        frazione = int(m.group(2)) / 100 if m.group(2) else 1.0
        d = json.loads(f.read_text(encoding="utf-8"))
        r = {
            "dataset": dataset,
            "braccio": m.group(1),
            "frazione": frazione,
            "seme": int(seme.replace("seed", "")),
            "ndcg": d["ndcg@10_completo"]["media"],
            "tempo_fit_s": d["tempo_fit_s"],
        }
        # quante celle addestrano davvero un modello additivo: e' la quota che spiega
        # perche' la qualita' regga, molto piu' di un'eventuale ridondanza delle coppie
        g = d.get("gam")
        if g:
            celle = g["gruppi_foglia_query"]
            motivi = g["motivi_ripiego"]
            r["celle"] = celle
            r["quota_con_gam"] = motivi.get("None", 0) / celle
            r["quota_fit_fallito"] = motivi.get("fit_fallito", 0) / celle
        righe.append(r)
    return righe


def aggrega(righe):
    """Media sui semi, per combinazione di dataset, braccio e frazione."""
    gruppi = defaultdict(list)
    for r in righe:
        gruppi[(r["dataset"], r["braccio"], r["frazione"])].append(r)
    fuori = {}
    for chiave, elenco in gruppi.items():
        ndcg = np.array([e["ndcg"] for e in elenco])
        tempo = np.array([e["tempo_fit_s"] for e in elenco])
        v = {"ndcg": ndcg.mean(), "ndcg_std": ndcg.std(),
             "tempo": tempo.mean(), "semi": len(elenco)}
        if "quota_con_gam" in elenco[0]:
            v["con_gam"] = np.mean([e["quota_con_gam"] for e in elenco])
            v["fit_fallito"] = np.mean([e["quota_fit_fallito"] for e in elenco])
        fuori[chiave] = v
    return fuori


def tabella(agg, uscita):
    with open(uscita, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "braccio", "frazione", "semi", "ndcg", "ndcg_std",
                    "tempo_fit_s", "ndcg_meno_pieno", "tempo_su_pieno",
                    "quota_celle_con_gam", "quota_celle_fit_fallito"])
        for dataset in DATASET:
            for braccio in BRACCI:
                pieno = agg.get((dataset, braccio, 1.0))
                if pieno is None:
                    continue
                for frazione in FRAZIONI:
                    v = agg.get((dataset, braccio, frazione))
                    if v is None:
                        continue
                    w.writerow([dataset, braccio, frazione, v["semi"],
                                round(v["ndcg"], 4), round(v["ndcg_std"], 4),
                                round(v["tempo"], 1),
                                round(v["ndcg"] - pieno["ndcg"], 4),
                                round(v["tempo"] / pieno["tempo"], 3),
                                round(v.get("con_gam", float("nan")), 3),
                                round(v.get("fit_fallito", float("nan")), 3)])
    print("scritto", uscita)


def grafico(agg, destinazioni):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SCURO, CHIARO = "#222222", "#888888"
    for cartella, lingua in destinazioni:
        cartella.mkdir(parents=True, exist_ok=True)
        fig, assi = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
        for ax, dataset in zip(assi, DATASET):
            for braccio, marker, colore in (("rtr", "s--", CHIARO), ("rtrwrulecard", "o-", SCURO)):
                x = [f for f in FRAZIONI if (dataset, braccio, f) in agg]
                y = [agg[(dataset, braccio, f)]["ndcg"] for f in x]
                e = [agg[(dataset, braccio, f)]["ndcg_std"] for f in x]
                ax.errorbar(x, y, yerr=e, fmt=marker, color=colore, capsize=2,
                            markersize=5, label=BRACCI[braccio])
            ax.set_title(DATASET[dataset], fontsize=10)
            ax.set_xscale("log")
            ax.set_xticks(FRAZIONI)
            ax.set_xticklabels(["0.1", "0.25", "0.5", "1.0"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linestyle=":", linewidth=0.6, color="#cccccc")
            ax.set_axisbelow(True)
        assi[0].set_ylabel("NDCG@10")
        etichetta = ("fraction of the documents used to build the pairs" if lingua == "en"
                     else "frazione dei documenti da cui si costruiscono le coppie")
        fig.supxlabel(etichetta, fontsize=9)
        assi[0].legend(frameon=False, loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(cartella / "sottocampionamento.pdf")
        plt.close(fig)
        print("scritta la figura in", cartella)


def main():
    a = sys.argv[1:]
    radice = Path(a[0]) if a else Path("risultati_subsample")
    righe = leggi(radice)
    print("run letti:", len(righe))
    agg = aggrega(righe)

    qui = Path(__file__).parent
    tabella(agg, qui / "sottocampionamento.csv")

    print()
    print(f"{'dataset':<11} {'braccio':<4} {'fraz.':>6} {'NDCG':>8} {'vs pieno':>9}"
          f" {'fit (s)':>9} {'costo':>7} {'celle con GAM':>14} {'fit fallito':>12}")
    for dataset in DATASET:
        for braccio in BRACCI:
            pieno = agg.get((dataset, braccio, 1.0))
            for frazione in FRAZIONI:
                v = agg.get((dataset, braccio, frazione))
                if v is None or pieno is None:
                    continue
                con = "%13.0f%%" % (v["con_gam"] * 100) if "con_gam" in v else " " * 14
                fal = "%11.0f%%" % (v["fit_fallito"] * 100) if "fit_fallito" in v else " " * 12
                print(f"{dataset:<11} {BRACCI[braccio]:<4} {frazione:>6.2f} {v['ndcg']:>8.4f}"
                      f" {v['ndcg'] - pieno['ndcg']:>+9.4f} {v['tempo']:>9.1f}"
                      f" {v['tempo'] / pieno['tempo']:>6.0%}{con}{fal}")

    if len(a) > 2:
        grafico(agg, [(Path(a[1]), "en"), (Path(a[2]), "it")])


if __name__ == "__main__":
    main()
