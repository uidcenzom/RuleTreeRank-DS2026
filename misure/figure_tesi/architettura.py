"""Lo schema dei due stadi, per il capitolo che descrive il framework.

I capitoli dal primo al quinto non avevano nessuna figura: l'architettura era
descritta solo a parole. Questo e' lo schema che fa capire il lavoro in dieci
secondi, e segue il percorso di un singolo documento dalla sua query al punteggio
finale.

Disegnata in grigio e nero come le altre figure della tesi, perche' si stampa anche
in bianco e nero, e in due lingue perche' le versioni sono due.

Uso:
  python architettura.py <cartella figure EN> <cartella figure IT>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCURO, MEDIO, CHIARO = "#222222", "#6b6b6b", "#b5b5b5"
SFONDO = "#f2f2f2"

TESTI = {
    "en": {
        "query": "query $q$\nand its documents",
        "stadio1": "First stage",
        "srt": "shallow rule tree\ndepth 5",
        "foglie": "leaves",
        "cella": "cell = leaf $\\times$ query",
        "stadio2": "Second stage, inside one cell",
        "coppie": "pairs of the cell\n$[\\mathbf{x}_a, \\mathbf{x}_b, |\\mathbf{x}_a - \\mathbf{x}_b|]$",
        "distanza": "distance model\nPDT or additive",
        "knn": "$k$ nearest\nneighbours",
        "residui": "mean of their\nresiduals",
        "punteggio": "$F(\\mathbf{x}) = r(\\mathbf{x}) + \\Delta(\\mathbf{x})$",
        "r": "coarse score $r(\\mathbf{x})$",
        "delta": "correction $\\Delta(\\mathbf{x})$",
        "sostituito": "this is the only\ncomponent we replace",
    },
    "it": {
        "query": "query $q$\ne i suoi documenti",
        "stadio1": "Primo stadio",
        "srt": "albero di regole\nprofondità 5",
        "foglie": "foglie",
        "cella": "cella = foglia $\\times$ query",
        "stadio2": "Secondo stadio, dentro una cella",
        "coppie": "coppie della cella\n$[\\mathbf{x}_a, \\mathbf{x}_b, |\\mathbf{x}_a - \\mathbf{x}_b|]$",
        "distanza": "modello di distanza\nPDT o additivo",
        "knn": "$k$ vicini\npiù prossimi",
        "residui": "media dei loro\nresidui",
        "punteggio": "$F(\\mathbf{x}) = r(\\mathbf{x}) + \\Delta(\\mathbf{x})$",
        "r": "punteggio grossolano $r(\\mathbf{x})$",
        "delta": "correzione $\\Delta(\\mathbf{x})$",
        "sostituito": "è l'unico componente\nche sostituiamo",
    },
}


def scatola(ax, x, y, w, h, testo, colore=SCURO, sfondo="white", spessore=1.0, size=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=spessore, edgecolor=colore, facecolor=sfondo))
    ax.text(x + w / 2, y + h / 2, testo, ha="center", va="center", fontsize=size, color=SCURO)


def freccia(ax, da, a, colore=MEDIO, stile="-|>"):
    ax.add_patch(FancyArrowPatch(da, a, arrowstyle=stile, mutation_scale=9,
                                 linewidth=0.9, color=colore, shrinkA=1, shrinkB=1))


def disegna(lingua, destinazione):
    T = TESTI[lingua]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ----- primo stadio -----
    ax.text(0.02, 0.955, T["stadio1"], fontsize=9, color=MEDIO, style="italic")
    scatola(ax, 0.03, 0.80, 0.22, 0.11, T["query"])
    scatola(ax, 0.33, 0.80, 0.24, 0.11, T["srt"])
    freccia(ax, (0.25, 0.855), (0.33, 0.855))

    # le tre foglie, quella centrale e' quella che seguiamo
    for i, x in enumerate((0.65, 0.76, 0.87)):
        evidenziata = i == 1
        scatola(ax, x, 0.805, 0.085, 0.10, "",
                colore=SCURO if evidenziata else CHIARO,
                sfondo=SFONDO if evidenziata else "white",
                spessore=1.2 if evidenziata else 0.8)
        freccia(ax, (0.57, 0.855), (x, 0.855), colore=MEDIO if evidenziata else CHIARO)
    ax.text(0.845, 0.765, T["foglie"], fontsize=8, color=MEDIO, ha="center")
    ax.text(0.8025, 0.93, T["cella"], fontsize=7.5, color=SCURO, ha="center")

    # ----- secondo stadio -----
    ax.add_patch(FancyBboxPatch((0.03, 0.17), 0.94, 0.45,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=0.8, edgecolor=CHIARO, facecolor="white",
                                linestyle=(0, (4, 3))))
    ax.text(0.05, 0.585, T["stadio2"], fontsize=9, color=MEDIO, style="italic")
    freccia(ax, (0.8025, 0.805), (0.8025, 0.62))

    scatola(ax, 0.06, 0.40, 0.25, 0.12, T["coppie"], size=7.5)
    scatola(ax, 0.39, 0.40, 0.23, 0.12, T["distanza"], sfondo=SFONDO, spessore=1.5)
    scatola(ax, 0.70, 0.40, 0.21, 0.12, T["knn"])
    scatola(ax, 0.70, 0.22, 0.21, 0.12, T["residui"])

    freccia(ax, (0.31, 0.46), (0.39, 0.46))
    freccia(ax, (0.62, 0.46), (0.70, 0.46))
    freccia(ax, (0.805, 0.40), (0.805, 0.34))

    ax.annotate(T["sostituito"], xy=(0.505, 0.40), xytext=(0.505, 0.255),
                ha="center", va="center", fontsize=7.5, color=SCURO,
                arrowprops=dict(arrowstyle="-|>", color=SCURO, lw=0.9,
                                shrinkA=2, shrinkB=2))

    # ----- punteggio finale -----
    scatola(ax, 0.33, 0.025, 0.34, 0.095, T["punteggio"], spessore=1.3)
    # la correzione esce dalla cella e scende al punteggio: l'etichetta sta sul tratto
    # verticale fuori dal riquadro tratteggiato, altrimenti ne tocca il bordo
    freccia(ax, (0.805, 0.22), (0.805, 0.14))
    freccia(ax, (0.805, 0.14), (0.67, 0.095), colore=MEDIO, stile="-")
    ax.text(0.825, 0.175, T["delta"], fontsize=7.5, color=MEDIO, ha="left")
    # r(x) viene dal primo stadio: scende lungo il margine sinistro invece di
    # attraversare lo schema in diagonale
    freccia(ax, (0.09, 0.80), (0.09, 0.075), colore=CHIARO, stile="-")
    freccia(ax, (0.09, 0.075), (0.33, 0.075), colore=CHIARO)
    ax.text(0.105, 0.115, T["r"], fontsize=7.5, color=MEDIO, ha="left")

    fig.tight_layout(pad=0.2)
    destinazione.mkdir(parents=True, exist_ok=True)
    fig.savefig(destinazione / "architettura.pdf")
    plt.close(fig)
    print("scritta architettura.pdf in", destinazione)


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        raise SystemExit("servono le due cartelle delle figure, inglese e italiana")
    disegna("en", Path(a[0]))
    disegna("it", Path(a[1]))


if __name__ == "__main__":
    main()
