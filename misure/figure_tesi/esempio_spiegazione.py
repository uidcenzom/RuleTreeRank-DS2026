"""Un esempio di spiegazione del modello additivo, con numeri veri.

La sezione che descrive le spiegazioni dice che il modello additivo produce una lista
di contributi con segno, uno per regola attivata, e poi non ne mostrava nessuna. Questa
figura mostra una spiegazione vera: si addestra una cella con la configurazione dei run,
si prende una coppia di candidati e si decompone la distanza nei contributi dei singoli
alberelli, ordinati per peso.

Nessun numero e' inventato: vengono tutti dal modello addestrato, e lo script li
ristampa a schermo cosi' si possono controllare.

Uso:
  python esempio_spiegazione.py <cartella figure EN> <cartella figure IT>
"""
import sys
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI.parent / "analisi"))
from interpretabilita import REPO, celle, contributi, crea_gam  # noqa: E402

sys.path.insert(0, str(REPO))
from ltr_utility.dataset import load_by_query_dataset, DatasetName  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCURO, MEDIO, CHIARO = "#222222", "#6b6b6b", "#b5b5b5"
# i nomi delle feature si leggono dal dataset, non si scrivono a mano: sono quelli
# della matrice di compatibilita' senza il prefisso `fitness_`

TESTI = {
    "en": {
        "titolo": "Distance between two candidates of the same cell",
        "contributo": "contribution to the distance",
        "base": "base value",
        "totale": "learned distance",
        "resto": "the other %d rules together",
        "regola": "rule",
    },
    "it": {
        "titolo": "Distanza fra due candidati della stessa cella",
        "contributo": "contributo alla distanza",
        "base": "valore base",
        "totale": "distanza appresa",
        "resto": "le altre %d regole insieme",
        "regola": "regola",
    },
}


def etichetta_regola(indici, nomi, lingua):
    """Da quali colonne della rappresentazione della coppia guarda questo alberello."""
    n = len(nomi)
    parti = []
    for i in indici:
        if i < n:
            parti.append(("cand. A " if lingua == "en" else "cand. A ") + nomi[i])
        elif i < 2 * n:
            parti.append(("cand. B " if lingua == "en" else "cand. B ") + nomi[i - n])
        else:
            parti.append(("|diff| " if lingua == "en" else "|diff| ") + nomi[i - 2 * n])
    return ", ".join(parti)


def scegli_cella(tutte, minimo=12, massimo=20):
    """Una cella abbastanza grande da essere interessante e abbastanza piccola da leggersi."""
    for query, foglia, C in tutte:
        if minimo <= C.shape[0] <= massimo:
            return query, foglia, C
    raise SystemExit("nessuna cella nella fascia richiesta")


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        raise SystemExit("servono le due cartelle delle figure, inglese e italiana")

    _, _, _, train_valid = load_by_query_dataset(REPO / "datasets", DatasetName["FINDHR"],
                                                 hold_out=(0.5, 0.2, 0.3), verbose=False)
    nomi = [str(c) for c in train_valid._columns_name]
    tutte = celle(train_valid, 5, 7)
    query, foglia, C = scegli_cella(tutte)
    print(f"cella scelta: query {query}, foglia {foglia}, {C.shape[0]} documenti")

    modello = crea_gam(4, 7)
    modello.fit(C)
    if modello.gam_ is None:
        raise SystemExit("la cella ha ripiegato sull'euclidea, scegline un'altra")

    # la coppia da spiegare: due candidati distinti, presi in modo riproducibile
    ia, ib = 0, 1
    c, feature = contributi(modello, C[[ia]], C[[ib]])
    c = c[0]
    base = float(modello.gam_.base_prediction_)
    distanza = float(modello.predict(C[[ia]], C[[ib]])[0])
    print(f"base {base:.3f}, somma contributi {c.sum():.3f}, distanza {distanza:.3f}")
    print(f"regole totali {len(c)}")

    ordine = np.argsort(np.abs(c))[::-1]
    quante = 5
    principali = ordine[:quante]
    resto = c[ordine[quante:]].sum()
    print("principali:", [(int(i), round(float(c[i]), 3)) for i in principali])
    print("resto:", round(float(resto), 3))

    for cartella, lingua in ((Path(a[0]), "en"), (Path(a[1]), "it")):
        T = TESTI[lingua]
        # il numero del round distingue regole che guardano la stessa feature
        etichette = [f"{T['regola']} {int(i) + 1}: " + etichetta_regola(feature[i], nomi, lingua)
                     for i in principali]
        etichette.append(T["resto"] % (len(c) - quante))
        valori = list(c[principali]) + [resto]

        fig, ax = plt.subplots(figsize=(6.4, 2.9))
        y = np.arange(len(valori))[::-1]
        colori = [SCURO if v >= 0 else MEDIO for v in valori]
        colori[-1] = CHIARO
        ax.barh(y, valori, color=colori, height=0.6)
        # spazio ai lati perche' le etichette dei valori non finiscano fuori
        limite = max(abs(min(valori)), abs(max(valori))) * 1.35
        ax.set_xlim(-limite, limite)
        for yi, v in zip(y, valori):
            ax.text(v + 0.02 * limite * (1 if v >= 0 else -1), yi, f"{v:+.2f}",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=7.5, color=SCURO)
        ax.set_yticks(y)
        ax.set_yticklabels(etichette, fontsize=7.5)
        ax.axvline(0, color=SCURO, linewidth=0.8)
        ax.set_xlabel(T["contributo"], fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", linestyle=":", linewidth=0.6, color="#cccccc")
        ax.set_axisbelow(True)
        # titolo e riepilogo stanno sopra, cosi' non si accavallano con l'asse
        ax.set_title(f"{T['titolo']}\n{T['base']} {base:+.2f}, {T['totale']} {distanza:.2f}",
                     fontsize=9, color=SCURO, linespacing=1.5)
        fig.tight_layout()
        cartella.mkdir(parents=True, exist_ok=True)
        fig.savefig(cartella / "esempio_spiegazione.pdf")
        plt.close(fig)
        print("scritta esempio_spiegazione.pdf in", cartella)


if __name__ == "__main__":
    main()
