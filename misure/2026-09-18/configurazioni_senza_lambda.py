"""
Configurazioni migliori senza lambda_scores, prese dalla model selection di Iommi.

I file *_query_model_selection.json mandati il 18 settembre 2026 contengono, per
ogni gruppo di query, tutte e 288 le configurazioni provate con il punteggio di
ognuna e l'indice di quella scelta. Il parametro lambda_scores non è
implementato nel codice che abbiamo, quindi per ogni gruppo prendiamo la
migliore fra le 144 che lo hanno spento: sugli stessi fold e sugli stessi gruppi
è esattamente il risultato che darebbe una model selection senza quel parametro.

Lo script scrive due file:
  configurazioni_senza_lambda.csv   una riga per gruppo, con la configurazione
                                    che sceglieremmo noi, quella scelta da lui e
                                    la differenza di punteggio
  configurazioni_senza_lambda.json  le configurazioni pronte per i run, raccolte
                                    per |phi|, con le query di ogni gruppo

Uso:
  python configurazioni_senza_lambda.py [cartella dei file di Iommi]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# il modello che ci interessa è Mix-RuleTreeRank, cioè il MixedRTR che usiamo;
# leggiamo anche RuleTreeRank perché serve come riferimento nelle tabelle
MODELLI = {
    "Mix-RuleTreeRank": "mix_rtr_query_model_selection.json",
    "RuleTreeRank": "rtr_query_model_selection.json",
}

# gli iperparametri che cambiano nella griglia, gli altri sono fissi
IPERPARAMETRI = ["feature_diff", "pdt_depth", "n_neighbors", "sdt_depth", "dist_objective"]


def scegli(gruppo: dict):
    """Configurazione scelta da lui e migliore fra quelle senza lambda_scores."""
    punteggi = np.asarray(gruppo["results"], dtype=float)
    configurazioni = gruppo["configs"]
    senza_lambda = np.array([not c.get("lambda_scores", False) for c in configurazioni])

    suo = int(gruppo["idx_best"])
    # fra le sole configurazioni senza lambda, la migliore per punteggio
    punteggi_ammessi = np.where(senza_lambda, punteggi, -np.inf)
    nostro = int(np.argmax(punteggi_ammessi))
    return suo, nostro, punteggi, senza_lambda, configurazioni


def analizza(percorso: Path, modello: str):
    dati = json.loads(percorso.read_text(encoding="utf-8"))
    righe, per_run, controlli = [], {}, {"idx_best non è il massimo": 0, "gruppi": 0}

    for gruppo in dati:
        suo, nostro, punteggi, senza_lambda, configurazioni = scegli(gruppo)
        phi = int(gruppo["qxm"])
        controlli["gruppi"] += 1
        if punteggi[suo] < punteggi.max() - 1e-12:
            controlli["idx_best non è il massimo"] += 1

        riga = {
            "modello": modello,
            "phi": phi,
            "gruppo": len(per_run.get(str(phi), [])),
            "n_query": len(gruppo["query"]),
            "ndcg_suo": round(float(punteggi[suo]), 4),
            "ndcg_senza_lambda": round(float(punteggi[nostro]), 4),
            "perdita": round(float(punteggi[suo] - punteggi[nostro]), 4),
            "lambda_acceso_nella_sua": bool(configurazioni[suo].get("lambda_scores", False)),
            "configurazione_cambia": suo != nostro,
        }
        for nome in IPERPARAMETRI:
            riga[nome] = configurazioni[nostro].get(nome)
            riga[f"{nome}_suo"] = configurazioni[suo].get(nome)
        righe.append(riga)

        scelta = {k: v for k, v in configurazioni[nostro].items() if k != "lambda_scores"}
        per_run.setdefault(str(phi), []).append({
            "gruppo": riga["gruppo"],
            "query": gruppo["query"],
            "ndcg_validazione": riga["ndcg_senza_lambda"],
            "configurazione": scelta,
        })

    return pd.DataFrame(righe), per_run, controlli


def riassunto(tabella: pd.DataFrame):
    """Quanto cambia, per |phi|, rinunciare a lambda_scores."""
    return (tabella.groupby(["modello", "phi"])
            .agg(gruppi=("gruppo", "count"),
                 lambda_acceso=("lambda_acceso_nella_sua", "sum"),
                 config_cambiata=("configurazione_cambia", "sum"),
                 perdita_media=("perdita", "mean"),
                 perdita_massima=("perdita", "max"))
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

    # le configurazioni ricorrenti, utili per restringere la griglia della GAM
    mix = tabella[tabella["modello"] == "Mix-RuleTreeRank"]
    print("\nvalori scelti più spesso senza lambda_scores (Mix-RuleTreeRank):")
    for nome in IPERPARAMETRI:
        conteggio = mix[nome].value_counts().head(4).to_dict()
        print(f"  {nome}: {conteggio}")


if __name__ == "__main__":
    main()
