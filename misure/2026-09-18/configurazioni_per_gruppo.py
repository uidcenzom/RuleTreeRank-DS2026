"""
La configurazione migliore per ogni gruppo di query, presa dalla model selection.

I file *_query_model_selection.json del gruppo contengono, per ogni gruppo di
query, tutte le configurazioni provate con il punteggio di ognuna e l'indice di
quella scelta. Da lì si ricava, senza addestrare niente, la configurazione
migliore per ciascun gruppo fra quelle che usano solo i parametri della nostra
griglia: sugli stessi fold e sugli stessi gruppi è il risultato che darebbe una
model selection fatta da noi su quella griglia.

Lo script scrive due file:
  configurazioni_per_gruppo.csv   una riga per gruppo, con la configurazione
                                  scelta da noi, quella del gruppo e la
                                  differenza di punteggio
  configurazioni_per_gruppo.json  le configurazioni pronte per i run, raccolte
                                  per |phi|, con le query di ogni gruppo

Uso:
  python configurazioni_per_gruppo.py [cartella dei file di model selection]
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

# gli iperparametri che cambiano nella griglia
IPERPARAMETRI = ["feature_diff", "pdt_depth", "n_neighbors", "sdt_depth", "dist_objective"]

# i parametri che la nostra implementazione conosce: tutti gli altri devono
# restare al valore neutro, altrimenti la configurazione non è riproducibile
NOSTRI = set(IPERPARAMETRI) | {
    "feature_concat", "feature_sq_diff", "subsample", "sdt_max_leaf_nodes",
    "min_samples_split", "verbose", "n_jobs_leaf",
}


def riproducibile(configurazione: dict) -> bool:
    """Vero se la configurazione usa solo parametri che sappiamo eseguire."""
    return not any(valore for chiave, valore in configurazione.items() if chiave not in NOSTRI)


def scegli(gruppo: dict):
    """Configurazione scelta dal gruppo e migliore fra quelle riproducibili."""
    punteggi = np.asarray(gruppo["results"], dtype=float)
    configurazioni = gruppo["configs"]
    ammesse = np.array([riproducibile(c) for c in configurazioni])

    loro = int(gruppo["idx_best"])
    punteggi_ammessi = np.where(ammesse, punteggi, -np.inf)
    nostro = int(np.argmax(punteggi_ammessi))
    return loro, nostro, punteggi, configurazioni


def analizza(percorso: Path, modello: str):
    dati = json.loads(percorso.read_text(encoding="utf-8"))
    righe, per_run, controlli = [], {}, {"idx_best non è il massimo": 0, "gruppi": 0}

    for gruppo in dati:
        loro, nostro, punteggi, configurazioni = scegli(gruppo)
        phi = int(gruppo["qxm"])
        controlli["gruppi"] += 1
        if punteggi[loro] < punteggi.max() - 1e-12:
            controlli["idx_best non è il massimo"] += 1

        riga = {
            "modello": modello,
            "phi": phi,
            "gruppo": len(per_run.get(str(phi), [])),
            "n_query": len(gruppo["query"]),
            "ndcg_del_gruppo": round(float(punteggi[loro]), 4),
            "ndcg_nostro": round(float(punteggi[nostro]), 4),
            "differenza": round(float(punteggi[loro] - punteggi[nostro]), 4),
            "configurazione_diversa": loro != nostro,
        }
        for nome in IPERPARAMETRI:
            riga[nome] = configurazioni[nostro].get(nome)
            riga[f"{nome}_del_gruppo"] = configurazioni[loro].get(nome)
        righe.append(riga)

        scelta = {k: v for k, v in configurazioni[nostro].items() if k in NOSTRI}
        per_run.setdefault(str(phi), []).append({
            "gruppo": riga["gruppo"],
            "query": gruppo["query"],
            "ndcg_validazione": riga["ndcg_nostro"],
            "configurazione": scelta,
        })

    return pd.DataFrame(righe), per_run, controlli


def riassunto(tabella: pd.DataFrame):
    """Per ogni |phi|, quanto si discosta la nostra scelta da quella del gruppo."""
    return (tabella.groupby(["modello", "phi"])
            .agg(gruppi=("gruppo", "count"),
                 config_diversa=("configurazione_diversa", "sum"),
                 differenza_media=("differenza", "mean"),
                 differenza_massima=("differenza", "max"))
            .round(4).reset_index())


def main():
    qui = Path(__file__).resolve()
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else qui.parents[2] / "dati_dal_gruppo" / "2026-09-18"

    tabelle, da_usare = [], {}
    for modello, nome_file in MODELLI.items():
        percorso = base / nome_file
        if not percorso.exists():
            print("manca", percorso)
            continue
        tabella, per_run, controlli = analizza(percorso, modello)
        tabelle.append(tabella)
        da_usare[modello] = per_run
        print(f"{modello}: {controlli['gruppi']} gruppi, "
              f"{controlli['idx_best non è il massimo']} con idx_best diverso dal massimo dei punteggi")

    if not tabelle:
        return
    tabella = pd.concat(tabelle, ignore_index=True)
    print()
    print(riassunto(tabella).to_string(index=False))

    csv = qui.with_suffix(".csv")
    tabella.to_csv(csv, index=False)
    js = qui.with_suffix(".json")
    js.write_text(json.dumps(da_usare, indent=1), encoding="utf-8")
    print("\nscritti", csv.name, "e", js.name)

    mix = tabella[tabella["modello"] == "Mix-RuleTreeRank"]
    print("\nvalori scelti più spesso (Mix-RuleTreeRank):")
    for nome in IPERPARAMETRI:
        print(f"  {nome}: {mix[nome].value_counts().head(4).to_dict()}")


if __name__ == "__main__":
    main()
