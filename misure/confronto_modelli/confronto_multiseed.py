"""
Confronto fra RTR con PDT e RTRwRuleCard su più seed.

Legge i risultati prodotti da scripts/esperimento.py (sul server o sul pc) e per
ogni dataset e |phi| riporta media e deviazione standard dell'NDCG@10 fra i seed,
la differenza fra i due modelli e un test di Wilcoxon appaiato per query, fatto
sui valori mediati fra i seed.

Uso:
  python confronto_multiseed.py <cartella results> [dataset ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PHI = [1, 2, 4, 6, 10]


def ndcg_medio(cartella: Path):
    """NDCG@10 complessivo di un run, oppure None se il run non c'è."""
    f = cartella / "metriche.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))["ndcg@10_completo"]["media"]


def per_query(cartella: Path):
    f = cartella / "ndcg_per_query.csv"
    if not f.exists():
        return None
    return pd.read_csv(f).set_index("query")["completo"]


def confronta(base: Path, dataset: str):
    righe = []
    for phi in PHI:
        valori = {"rtr": [], "rtrwrulecard": []}
        query = {"rtr": [], "rtrwrulecard": []}
        for variante in valori:
            for seed_dir in sorted((base / dataset / variante / f"phi{phi}").glob("seed*")):
                v = ndcg_medio(seed_dir)
                if v is not None:
                    valori[variante].append(v)
                    q = per_query(seed_dir)
                    if q is not None:
                        query[variante].append(q)
        if not valori["rtr"] or not valori["rtrwrulecard"]:
            continue
        pdt, gam = np.array(valori["rtr"]), np.array(valori["rtrwrulecard"])
        riga = {
            "dataset": dataset, "phi": phi,
            "seed_pdt": len(pdt), "seed_gam": len(gam),
            "pdt_media": round(pdt.mean(), 4), "pdt_std": round(pdt.std(ddof=1), 4) if len(pdt) > 1 else 0.0,
            "gam_media": round(gam.mean(), 4), "gam_std": round(gam.std(ddof=1), 4) if len(gam) > 1 else 0.0,
            "differenza": round(gam.mean() - pdt.mean(), 4),
        }
        if query["rtr"] and query["rtrwrulecard"]:
            # media fra i seed dell'NDCG di ogni query, poi test appaiato sulle query
            qp = pd.concat(query["rtr"], axis=1).mean(axis=1)
            qg = pd.concat(query["rtrwrulecard"], axis=1).mean(axis=1)
            d = (qg - qp).dropna()
            d = d[d != 0]
            riga["query_meglio_gam"] = int((d > 0).sum())
            riga["query_meglio_pdt"] = int((d < 0).sum())
            riga["p_wilcoxon"] = round(wilcoxon(d).pvalue, 4) if len(d) > 0 else None
        righe.append(riga)
    return righe


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    dataset = sys.argv[2:] or ["FINDHRLIST", "FINDHR"]
    righe = [r for d in dataset for r in confronta(base, d)]
    if not righe:
        print("nessun risultato trovato in", base)
        return
    tabella = pd.DataFrame(righe)
    print(tabella.to_string(index=False))
    out = Path(__file__).with_suffix(".csv")
    tabella.to_csv(out, index=False)
    print("\nscritto", out)


if __name__ == "__main__":
    main()
