"""
Confronto fra RTR con PDT e RTRwRuleCard su FINDHR list, per ogni |phi|.

Legge i risultati salvati da scripts/esperimento.py (seed 0) e riporta NDCG@10,
tempi, statistiche della GAM e il test di Wilcoxon appaiato sull'NDCG@10 delle
singole query. Il test considera solo le query in cui i due modelli differiscono.
"""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

R = Path(r"C:\Users\manzo\Desktop\fork\RuleTreeRank-DS2026\results\FINDHRLIST")


def metriche(variante, phi):
    return json.loads((R / variante / f"phi{phi}" / "seed0" / "metriche.json").read_text(encoding="utf-8"))


righe = []
for phi in [1, 2, 4, 6, 10]:
    pdt, gam = metriche("rtr", phi), metriche("rtrwrulecard", phi)
    q_pdt = pd.read_csv(R / "rtr" / f"phi{phi}" / "seed0" / "ndcg_per_query.csv").set_index("query")["completo"]
    q_gam = pd.read_csv(R / "rtrwrulecard" / f"phi{phi}" / "seed0" / "ndcg_per_query.csv").set_index("query")["completo"]
    diff = q_gam - q_pdt
    termini = gam["gam"]["quota_termini_per_colonna"]
    righe.append({
        "phi": phi,
        "ndcg_pdt": round(pdt["ndcg@10_completo"]["media"], 4),
        "ndcg_gam": round(gam["ndcg@10_completo"]["media"], 4),
        "differenza": round(gam["ndcg@10_completo"]["media"] - pdt["ndcg@10_completo"]["media"], 4),
        "ndcg_knn_euclideo": round(gam["ndcg@10_knn_euclideo"]["media"], 4),
        "ndcg_solo_r": round(gam["ndcg@10_solo_r"]["media"], 4),
        "query_meglio_gam": int((diff > 0).sum()),
        "query_meglio_pdt": int((diff < 0).sum()),
        "query_uguali": int((diff == 0).sum()),
        "p_wilcoxon": round(wilcoxon(diff[diff != 0]).pvalue, 3),
        "fit_pdt_s": round(pdt["tempo_fit_s"], 1),
        "fit_gam_min": round(gam["tempo_fit_s"] / 60, 1),
        "gam_quota_round_al_massimo": round(gam["gam"]["quota_round_al_massimo"], 2),
        "gam_termini_a": round(termini.get("a", 0), 2),
        "gam_termini_b": round(termini.get("b", 0), 2),
        "gam_termini_differenza": round(termini.get("differenza", 0), 2),
    })

tabella = pd.DataFrame(righe)
print(tabella.to_string(index=False))
tabella.to_csv(Path(__file__).with_suffix(".csv"), index=False)
