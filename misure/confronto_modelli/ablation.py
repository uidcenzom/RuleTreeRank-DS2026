"""
Le quattro varianti della figura di ablation del paper, con i nostri due modelli.

Il modello completo corregge il punteggio del primo stadio guardando i documenti
simili dentro la stessa cella, dove una cella e' l'incrocio fra una foglia e un
gruppo di query. Le varianti tolgono un pezzo per volta:

  completo        primo stadio + correzione dentro la cella
  senza query     un modello di distanza per foglia e non per cella, quindi
                  documenti di query diverse si fanno da vicini a vicenda
  solo r(x)       solo il primo stadio, nessuna correzione
  solo s(x)       nessuna foglia, la correzione lavora su tutto il gruppo

Le prime tre si leggono dagli stessi run, perche' ogni esperimento salva anche le
predizioni del solo primo stadio. La quarta e la seconda hanno run dedicati.

Uso:
  python ablation.py <cartella results> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PHI = [1, 2, 4, 6, 10]
DATASET = ["FINDHRLIST", "FINDHR"]
DISTANZE = {"PDT": "rtr", "GAM": "rtrwrulecard"}


def valori(cartelle, dataset, variante, predizione="completo"):
    """Media e numero di seed di un run, cercando in piu' cartelle senza contare due volte."""
    trovati, visti = [], set()
    for base in cartelle:
        for d in sorted(Path(base).glob(f"{dataset}/{variante}/phi*/seed*")):
            chiave = (d.parts[-2], d.parts[-1])
            f = d / "metriche.json"
            if chiave in visti or not f.exists():
                continue
            visti.add(chiave)
            m = json.loads(f.read_text(encoding="utf-8"))
            voce = m.get(f"ndcg@10_{predizione}")
            if voce:
                trovati.append((int(d.parts[-2][3:]), voce["media"]))
    return trovati


def media_per_phi(coppie, phi):
    v = [x for p, x in coppie if p == phi]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def tabella(cartelle):
    righe = []
    for dataset in DATASET:
        for nome, variante in DISTANZE.items():
            completo = valori(cartelle, dataset, variante)
            solo_r = valori(cartelle, dataset, variante, "solo_r")
            senza_query = valori(cartelle, dataset, f"{variante}_senzaquery")
            solo_s = valori(cartelle, dataset, f"{variante}_senzafoglie")
            for phi in PHI:
                riga = {"dataset": dataset, "distanza": nome, "phi": phi}
                for etichetta, coppie in [("completo", completo), ("senza query", senza_query),
                                          ("solo r(x)", solo_r), ("solo s(x)", solo_s)]:
                    media, n = media_per_phi(coppie, phi)
                    riga[etichetta] = round(media, 4) if media is not None else None
                    riga[f"seed {etichetta}"] = n
                righe.append(riga)
    return pd.DataFrame(righe)


def main():
    cartelle = sys.argv[1:] or ["results_server", "results"]
    t = tabella(cartelle)
    colonne = ["dataset", "distanza", "phi", "completo", "senza query", "solo r(x)", "solo s(x)"]
    print(t[colonne].to_string(index=False))

    print("\nquanto costa togliere un pezzo, in media:")
    for etichetta in ["senza query", "solo r(x)", "solo s(x)"]:
        d = (t[etichetta] - t["completo"]).dropna()
        if len(d):
            print(f"  {etichetta:12} {d.mean():+.4f}  (da {d.min():+.4f} a {d.max():+.4f})")

    print("\nseed disponibili per colonna:",
          {c: sorted(set(t[f"seed {c}"])) for c in ["completo", "senza query", "solo r(x)", "solo s(x)"]})

    out = Path(__file__).with_suffix(".csv")
    t.to_csv(out, index=False)
    print("\nscritto", out)


if __name__ == "__main__":
    main()
