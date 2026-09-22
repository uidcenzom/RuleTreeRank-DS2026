"""
Le varianti provate sul primo e sul secondo stadio, contro RTR con il PDT.

Il riferimento e' sempre `rtr`, cioe' il modello completo con il PDT come
distanza e il kNN uniforme come aggregatore. Ogni variante cambia un pezzo solo:

  kNN pesato        l'aggregatore pesa i vicini per 1/distanza invece che in modo
                    uniforme: con la media uniforme contano solo *quali* vicini si
                    scelgono, con i pesi contano anche i valori della distanza
  minimo per foglia un vincolo sul numero di documenti di training in ogni foglia
                    del primo stadio, che rende le celle piu' grandi
  foresta nella cella  l'aggregatore diventa una foresta che guarda direttamente le
                    feature: non e' interpretabile, sta qui come upper bound

I run dei semi stanno in cartelle diverse (il primo seme e' stato pubblicato,
gli altri no), quindi lo script accetta piu' radici e le unisce senza contare due
volte lo stesso seme.

Il test e' un Wilcoxon appaiato sulle query, fatto sull'NDCG per query mediato
fra i semi, come negli altri confronti.

Uso:
  python varianti_del_modello.py <cartella> [altra cartella ...]
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

PHI = [1, 2, 4, 6, 10]
DATASET = ["FINDHR", "FINDHRLIST"]
RIFERIMENTO = "rtr"
VARIANTI = {
    "rtr_pesato": "kNN pesato per distanza",
    "rtr_minfoglia6": "minimo 6 documenti per foglia",
    "rtr_minfoglia10": "minimo 10 documenti per foglia",
    "rtr_forestainfoglia": "foresta nella cella (non interpretabile)",
}


def per_query(cartelle, dataset, variante, phi):
    """NDCG per query mediato sui semi, cercando in piu' cartelle senza duplicare."""
    serie, visti = [], set()
    for base in cartelle:
        for d in sorted(Path(base).glob(f"{dataset}/{variante}/phi{phi}/seed*")):
            f = d / "ndcg_per_query.csv"
            if d.name in visti or not f.exists():
                continue
            visti.add(d.name)
            serie.append(pd.read_csv(f).set_index("query")["completo"])
    if not serie:
        return None, 0
    return pd.concat(serie, axis=1).mean(axis=1), len(serie)


def main():
    cartelle = sys.argv[1:] or ["risultati_semi", "risultati"]

    righe = []
    for dataset in DATASET:
        for phi in PHI:
            base, semi_base = per_query(cartelle, dataset, RIFERIMENTO, phi)
            if base is None:
                continue
            for variante, descrizione in VARIANTI.items():
                v, semi = per_query(cartelle, dataset, variante, phi)
                if v is None:
                    continue
                d = (v - base).dropna()
                d = d[d != 0]
                righe.append({
                    "dataset": dataset,
                    "phi": phi,
                    "variante": descrizione,
                    "semi": semi,
                    "semi_riferimento": semi_base,
                    "riferimento": round(float(base.mean()), 4),
                    "con_la_variante": round(float(v.mean()), 4),
                    "differenza": round(float(v.mean() - base.mean()), 4),
                    "p": round(float(wilcoxon(d).pvalue), 4) if len(d) else None,
                })

    if not righe:
        print("nessun run trovato nelle cartelle indicate")
        return

    t = pd.DataFrame(righe)
    print("=== ogni variante contro RTR con il PDT ===")
    print(t.to_string(index=False))

    print("\n=== riassunto per variante ===")
    for descrizione, sotto in t.groupby("variante", sort=False):
        positive = int((sotto["differenza"] > 0).sum())
        significative = int((sotto["p"] < 0.05).sum())
        semi = sorted(set(sotto["semi"]))
        print(f"  {descrizione}: media {sotto['differenza'].mean():+.4f}, "
              f"positiva in {positive} su {len(sotto)}, "
              f"significativa in {significative} su {len(sotto)}, "
              f"semi {semi}")

    print("\nIl solo cambio di seme sposta l'NDCG di circa 0.0035: una differenza")
    print("piu' piccola di cosi' non si distingue dal rumore.")

    uscita = Path(__file__).with_suffix(".csv")
    t.to_csv(uscita, index=False)
    print("\nscritto", uscita.name)


if __name__ == "__main__":
    main()
