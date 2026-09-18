"""
Perché la GAM guadagna più su FINDHR che su FINDHRl.

La quota di documenti su cui la distanza può incidere è 26-42% su FINDHR e
84-90% su FINDHRl, ma il vantaggio della GAM sul PDT è più grande su FINDHR.
Lo script misura tre cose sugli stessi risultati, per capire perché:

- quanti documenti cambiano score fra i due modelli e di quanto;
- lo spostamento medio rapportato alla deviazione standard dei voti, cioè alla
  scala che conta per l'NDCG;
- quanti documenti cambiano posizione nel ranking della loro query, di quante
  posizioni, e quanti fra quelli nei primi dieci.

Uso:
  python perche_findhr_guadagna_di_piu.py <cartella results>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PHI = [1, 2, 4, 6, 10]
SEED = range(5)


def posizioni(pred, q):
    """Posizione di ogni documento dentro la sua query, 0 per il primo."""
    pos = np.empty(len(pred), dtype=int)
    for query in np.unique(q):
        m = q == query
        ordine = np.argsort(-pred[m], kind="stable")
        r = np.empty(m.sum(), dtype=int)
        r[ordine] = np.arange(m.sum())
        pos[m] = r
    return pos


def analizza(base: Path, dataset: str):
    righe = []
    for phi in PHI:
        misure = {k: [] for k in ["diversi", "salto_score", "salto_relativo", "cambia_posizione",
                                  "posizioni_spostate", "top10_cambia", "livelli_voto"]}
        for seed in SEED:
            f_pdt = base / dataset / "rtr" / f"phi{phi}" / f"seed{seed}" / "predizioni.npz"
            f_gam = base / dataset / "rtrwrulecard" / f"phi{phi}" / f"seed{seed}" / "predizioni.npz"
            if not f_pdt.exists() or not f_gam.exists():
                continue
            pdt, gam = np.load(f_pdt), np.load(f_gam)
            q, y = pdt["query"], pdt["etichette"]
            scarto = np.abs(gam["completo"] - pdt["completo"])
            cambiati = scarto > 1e-9
            misure["diversi"].append(cambiati.mean())
            misure["salto_score"].append(scarto[cambiati].mean() if cambiati.any() else 0.0)
            misure["salto_relativo"].append(scarto[cambiati].mean() / y.std() if cambiati.any() else 0.0)
            pos_pdt, pos_gam = posizioni(pdt["completo"], q), posizioni(gam["completo"], q)
            spostati = pos_pdt != pos_gam
            misure["cambia_posizione"].append(spostati.mean())
            misure["posizioni_spostate"].append(np.abs(pos_pdt - pos_gam)[spostati].mean() if spostati.any() else 0.0)
            primi = pos_pdt < 10
            misure["top10_cambia"].append((primi & spostati).sum() / max(primi.sum(), 1))
            misure["livelli_voto"].append(len(np.unique(y)))
        if misure["diversi"]:
            righe.append({
                "dataset": dataset, "phi": phi,
                "livelli_voto": int(np.mean(misure["livelli_voto"])),
                "doc_score_diverso": round(float(np.mean(misure["diversi"])), 3),
                "salto_score": round(float(np.mean(misure["salto_score"])), 4),
                "salto_su_devstd_voti": round(float(np.mean(misure["salto_relativo"])), 3),
                "doc_cambia_posizione": round(float(np.mean(misure["cambia_posizione"])), 3),
                "posizioni_spostate": round(float(np.mean(misure["posizioni_spostate"])), 2),
                "top10_cambia": round(float(np.mean(misure["top10_cambia"])), 3),
            })
    return righe


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results_server")
    righe = [r for d in ["FINDHR", "FINDHRLIST"] for r in analizza(base, d)]
    tabella = pd.DataFrame(righe)
    print(tabella.to_string(index=False))
    out = Path(__file__).with_suffix(".csv")
    tabella.to_csv(out, index=False)
    print("\nscritto", out)


if __name__ == "__main__":
    main()
