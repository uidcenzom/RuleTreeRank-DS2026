"""
Quanto pesa ogni iperparametro, dai file di model selection di Iommi.

I suoi file danno il punteggio di tutte le configurazioni su ognuno dei 218
gruppi di query, non solo di quella scelta. Sono quindi una griglia completa già
pagata, da cui si legge senza far girare niente:

1. l'effetto medio di ogni valore di ogni iperparametro, calcolato dentro il
   gruppo per togliere di mezzo il fatto che certi gruppi sono più facili;
2. quanto costa fissare un iperparametro per tutti i gruppi invece di sceglierlo
   gruppo per gruppo, cioè quanto vale davvero cercarlo;
3. quanto vale scegliere la configurazione gruppo per gruppo invece di una sola
   configurazione buona per tutti: è la premessa dell'intero impianto a gruppi di
   query, e nessuno l'aveva ancora misurata.

Si lavora sulle configurazioni che usano solo i parametri della nostra griglia,
cioè quelle che possiamo eseguire.

Uso:
  python peso_degli_iperparametri.py [cartella dei file di Iommi]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODELLI = {
    "Mix-RuleTreeRank": "mix_rtr_query_model_selection.json",
    "RuleTreeRank": "rtr_query_model_selection.json",
}
IPERPARAMETRI = ["feature_diff", "pdt_depth", "n_neighbors", "sdt_depth", "dist_objective"]

# i parametri che la nostra implementazione conosce: tutti gli altri devono
# restare al valore neutro, altrimenti la configurazione non è riproducibile
NOSTRI = set(IPERPARAMETRI) | {
    "feature_concat", "feature_sq_diff", "subsample", "sdt_max_leaf_nodes",
    "min_samples_split", "verbose", "n_jobs_leaf",
}


def tabella_lunga(percorso: Path, modello: str) -> pd.DataFrame:
    """Una riga per (gruppo, configurazione), con gli iperparametri e il punteggio."""
    righe = []
    for numero, gruppo in enumerate(json.loads(percorso.read_text(encoding="utf-8"))):
        for conf, punteggio in zip(gruppo["configs"], gruppo["results"]):
            if any(v for k, v in conf.items() if k not in NOSTRI):
                continue
            righe.append({"modello": modello, "phi": int(gruppo["qxm"]), "gruppo": numero,
                          **{k: conf.get(k) for k in IPERPARAMETRI}, "ndcg": float(punteggio)})
    t = pd.DataFrame(righe)
    # dentro ogni gruppo tolgo il livello medio: restano le differenze fra configurazioni
    t["scarto"] = t["ndcg"] - t.groupby(["modello", "phi", "gruppo"])["ndcg"].transform("mean")
    return t


def effetti(t: pd.DataFrame) -> pd.DataFrame:
    """Effetto medio di ogni valore, in punti di NDCG rispetto alla media del gruppo."""
    righe = []
    for nome in IPERPARAMETRI:
        for valore, sotto in t.groupby(nome):
            righe.append({"iperparametro": nome, "valore": valore,
                          "effetto": round(sotto["scarto"].mean(), 4),
                          "configurazioni": len(sotto)})
    return pd.DataFrame(righe).sort_values(["iperparametro", "effetto"], ascending=[True, False])


def costo_di_fissarlo(t: pd.DataFrame) -> pd.DataFrame:
    """Quanto si perde a tenere un iperparametro fisso invece di sceglierlo per gruppo.

    Per ogni gruppo: il meglio ottenibile con tutte le configurazioni, meno il
    meglio ottenibile tenendo l'iperparametro a un valore solo. Si riporta il
    valore che costa meno, cioè il migliore da fissare se se ne deve fissare uno.
    """
    chiavi = ["modello", "phi", "gruppo"]
    massimo = t.groupby(chiavi)["ndcg"].max().rename("migliore")
    righe = []
    for nome in IPERPARAMETRI:
        per_valore = t.groupby(chiavi + [nome])["ndcg"].max().rename("migliore_con_valore").reset_index()
        unito = per_valore.merge(massimo, on=chiavi)
        unito["perdita"] = unito["migliore"] - unito["migliore_con_valore"]
        media = unito.groupby(["modello", nome])["perdita"].mean().reset_index()
        for modello, sotto in media.groupby("modello"):
            riga = sotto.loc[sotto["perdita"].idxmin()]
            righe.append({"modello": modello, "iperparametro": nome,
                          "valore_migliore_da_fissare": riga[nome],
                          "perdita_media": round(float(riga["perdita"]), 4),
                          "perdita_peggior_valore": round(float(sotto["perdita"].max()), 4)})
    return pd.DataFrame(righe)


def valore_della_scelta_per_gruppo(t: pd.DataFrame) -> pd.DataFrame:
    """Scegliere per gruppo contro usare un'unica configurazione per tutti.

    La configurazione unica è la migliore in media sui gruppi di quel |phi|, cioè
    quella che sceglierebbe chi non facesse la model selection per gruppo.
    """
    righe = []
    for (modello, phi), sotto in t.groupby(["modello", "phi"]):
        per_gruppo = sotto.groupby("gruppo")["ndcg"].max().mean()
        chiavi_conf = IPERPARAMETRI
        media_per_conf = sotto.groupby(chiavi_conf)["ndcg"].mean()
        unica = media_per_conf.max()
        righe.append({"modello": modello, "phi": phi,
                      "scelta_per_gruppo": round(float(per_gruppo), 4),
                      "configurazione_unica": round(float(unica), 4),
                      "guadagno": round(float(per_gruppo - unica), 4)})
    return pd.DataFrame(righe)


def main():
    qui = Path(__file__).resolve()
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else qui.parents[2] / "dati_dal_gruppo" / "2026-09-18"

    tabelle = []
    for modello, nome_file in MODELLI.items():
        percorso = base / nome_file
        if percorso.exists():
            tabelle.append(tabella_lunga(percorso, modello))
        else:
            print("manca", percorso)
    if not tabelle:
        return
    t = pd.concat(tabelle, ignore_index=True)
    print(f"{len(t)} righe: {t['gruppo'].nunique()} gruppi per |phi|, "
          f"{len(t) // t.groupby(['modello', 'phi', 'gruppo']).ngroups} configurazioni per gruppo\n")

    mix = t[t["modello"] == "Mix-RuleTreeRank"]
    print("=== effetto medio di ogni valore, Mix-RuleTreeRank (punti di NDCG sulla media del gruppo) ===")
    print(effetti(mix).to_string(index=False))

    print("\n=== quanto costa fissare un iperparametro invece di sceglierlo per gruppo ===")
    print(costo_di_fissarlo(t).to_string(index=False))

    print("\n=== quanto vale scegliere per gruppo invece di una configurazione unica ===")
    scelta = valore_della_scelta_per_gruppo(t)
    print(scelta.to_string(index=False))

    effetti(mix).to_csv(qui.with_name("peso_degli_iperparametri_effetti.csv"), index=False)
    costo_di_fissarlo(t).to_csv(qui.with_name("peso_degli_iperparametri_costo.csv"), index=False)
    scelta.to_csv(qui.with_name("peso_degli_iperparametri_scelta_per_gruppo.csv"), index=False)
    print("\nscritti i tre csv in", qui.parent.name)


if __name__ == "__main__":
    main()
