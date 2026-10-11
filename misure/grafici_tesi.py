"""
I tre grafici del capitolo sperimentale della tesi.

Legge le stesse tabelle da cui vengono i numeri del capitolo e produce tre PDF
vettoriali, uno per figura. Non ricalcola niente: se una tabella cambia, basta
rilanciare questo script e le figure si aggiornano.

I file vengono scritti nella cartella `figures/` di entrambe le versioni della
tesi, quella inglese e quella italiana, con le didascalie degli assi nella lingua
corrispondente. I nomi dei file sono gli stessi nelle due versioni, così che il
sorgente dei capitoli resti identico.

Uso:
  python grafici_tesi.py <cartella con le tabelle> <cartella tesi EN> <cartella tesi IT>
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHI = [1, 2, 4, 6, 10]
# grigi e nero: la tesi si stampa anche in bianco e nero
SCURO, CHIARO = "#222222", "#888888"


def valore(s):
    """Le celle della scala dei modelli sono scritte come media piu' o meno deviazione."""
    return float(str(s).split("±")[0].strip())


def stile(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, color="#cccccc")
    ax.set_axisbelow(True)


def guadagno_per_phi(tabelle, uscita, lingua):
    """Figura 1: il guadagno della GAM sul PDT al variare del numero di query."""
    righe = list(csv.DictReader(open(tabelle / "confronto_multiseed_server.csv", encoding="utf-8")))
    dati = {d: {int(r["phi"]): float(r["gain"]) for r in righe if r["dataset"] == d}
            for d in ("FINDHR", "FINDHRLIST")}

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(PHI, [dati["FINDHR"][p] for p in PHI], "o-", color=SCURO, label="FindHR")
    ax.plot(PHI, [dati["FINDHRLIST"][p] for p in PHI], "s--", color=CHIARO,
            label="FindHR$_\\ell$")
    ax.axhline(0.00225, color="#bb0000", linestyle="-.", linewidth=1)
    testo = ("95% of null comparisons" if lingua == "en" else "95% dei confronti nulli")
    ax.annotate(testo, xy=(1, 0.00225), xytext=(2, 5), textcoords="offset points",
                ha="left", va="bottom", fontsize=8, color="#bb0000")

    ax.set_xlabel("queries per model $|\\varphi|$" if lingua == "en"
                  else "query per modello $|\\varphi|$")
    ax.set_ylabel("gain in NDCG@10" if lingua == "en" else "guadagno di NDCG@10")
    ax.set_xticks(PHI)
    ax.set_ylim(0, 0.009)
    ax.legend(frameon=False, loc="upper left")
    stile(ax)
    fig.tight_layout()
    fig.savefig(uscita / "guadagno_per_phi.pdf")
    plt.close(fig)


def intervallo_riferimento(tabelle, uscita, lingua):
    """Figura 2: dove sta il framework fra l'ordinamento casuale e il soffitto."""
    righe = list(csv.DictReader(open(tabelle / "scala_dei_modelli.csv", encoding="utf-8")))
    colonne = [
        ("casuale", "random ordering", "ordinamento casuale", False),
        ("solo r(x)", "first stage only", "solo primo stadio", True),
        ("euclidea in foglia", "Euclidean $k$-NN", "$k$-NN euclideo", True),
        ("RTR (PDT)", "RTR (PDT)", "RTR (PDT)", True),
        ("RTRwRuleCard (GAM)", "RTRwRuleCard", "RTRwRuleCard", True),
        ("foresta in foglia", "forest in cell", "foresta nella cella", False),
        ("boosting sul gruppo", "boosting on group", "boosting sul gruppo", False),
        ("LambdaMART", "LambdaMART", "LambdaMART", False),
    ]
    nomi = [c[1] if lingua == "en" else c[2] for c in colonne]
    minimi = [min(valore(r[c[0]]) for r in righe) for c in colonne]
    massimi = [max(valore(r[c[0]]) for r in righe) for c in colonne]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (lo, hi, c) in enumerate(zip(minimi, massimi, colonne)):
        colore = SCURO if c[3] else CHIARO
        ax.plot([lo, hi], [i, i], color=colore, linewidth=3, solid_capstyle="round")
        ax.plot([lo, hi], [i, i], "|", color=colore, markersize=7)
    ax.set_yticks(range(len(nomi)))
    ax.set_yticklabels(nomi, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("NDCG@10")
    ax.set_xlim(0.50, 1.0)
    etichetta = ("interpretable in black" if lingua == "en"
                 else "in nero i modelli interpretabili")
    ax.annotate(etichetta, xy=(0.52, len(nomi) - 0.4), fontsize=8, color=SCURO)
    stile(ax)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, color="#cccccc")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(uscita / "intervallo_riferimento.pdf")
    plt.close(fig)


def costo_contro_guadagno(tabelle, uscita, lingua):
    """Figura 3: quanto costa il guadagno, combinazione per combinazione."""
    righe = list(csv.DictReader(open(tabelle / "confronto_multiseed_server.csv", encoding="utf-8")))
    gain = {(r["dataset"], int(r["phi"])): float(r["gain"]) for r in righe}
    # i rapporti di costo vengono dalla stessa tabella del capitolo, non riscritti a mano
    costi = list(csv.DictReader(open(tabelle / "costi.csv", encoding="utf-8")))
    rapporti = {(r["dataset"], int(r["phi"])): float(r["rapporto"]) for r in costi}

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for ds, marker, colore, nome in (("FINDHR", "o", SCURO, "FindHR"),
                                     ("FINDHRLIST", "s", CHIARO, "FindHR$_\\ell$")):
        x = [rapporti[(ds, p)] for p in PHI]
        y = [gain[(ds, p)] for p in PHI]
        ax.scatter(x, y, marker=marker, color=colore, s=38, label=nome, zorder=3)
        for p, xi, yi in zip(PHI, x, y):
            ax.annotate(str(p), (xi, yi), textcoords="offset points", xytext=(5, -3),
                        fontsize=7, color=colore)
    ax.set_xlabel("training cost, times the PDT" if lingua == "en"
                  else "costo di addestramento, volte il PDT")
    ax.set_ylabel("gain in NDCG@10" if lingua == "en" else "guadagno di NDCG@10")
    ax.set_ylim(0, 0.009)
    ax.legend(frameon=False, loc="upper left")
    stile(ax)
    fig.tight_layout()
    fig.savefig(uscita / "costo_contro_guadagno.pdf")
    plt.close(fig)


def main():
    a = sys.argv[1:]
    tabelle = Path(a[0]) if a else Path("confronto_modelli")
    destinazioni = [(Path(a[1]), "en"), (Path(a[2]), "it")] if len(a) > 2 else []
    for cartella, lingua in destinazioni:
        cartella.mkdir(parents=True, exist_ok=True)
        guadagno_per_phi(tabelle, cartella, lingua)
        intervallo_riferimento(tabelle, cartella, lingua)
        costo_contro_guadagno(tabelle, cartella, lingua)
        print("scritte 3 figure in", cartella)


if __name__ == "__main__":
    main()
