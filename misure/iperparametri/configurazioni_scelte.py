"""
Le configurazioni scelte dalla nostra model selection, e il confronto con le loro.

Legge i file prodotti da `scripts/model_selection.py`, cioè una voce per gruppo di
query con tutte le configurazioni provate, il punteggio di ognuna e l'indice della
migliore, e produce due cose:

1. il riepilogo di cosa è stato scelto, per dataset, modello e |phi|: quali valori
   ricorrono, quanto vale il punteggio di validazione, quanto è distante la
   migliore dalla peggiore dentro lo stesso gruppo (cioè quanto è servito cercare);
2. il file pronto per i run finali, con la configurazione di ogni gruppo.

Se nella cartella indicata con --loro ci sono i file di model selection del gruppo
di ricerca, fa anche il controllo incrociato sul braccio con il PDT: le
configurazioni che la nostra procedura sceglie dovrebbero coincidere con quelle
che si ricavano dai loro file, perché la griglia, i gruppi e la convalida sono gli
stessi. Se non coincidono, è un segnale che qualcosa nella procedura differisce, ed
è meglio saperlo prima di scrivere i risultati.

Uso:
  python configurazioni_scelte.py <cartella model_selection> [--loro <cartella>]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

IPERPARAMETRI = ["feature_diff", "pdt_depth", "n_neighbors", "sdt_depth", "dist_objective"]
NOSTRI_FILE = {("FINDHR", "rtr"): "FINDHR_rtr.json",
               ("FINDHR", "rtrwrulecard"): "FINDHR_rtrwrulecard.json",
               ("FINDHRLIST", "rtr"): "FINDHRLIST_rtr.json",
               ("FINDHRLIST", "rtrwrulecard"): "FINDHRLIST_rtrwrulecard.json"}
LORO_FILE = "rtr_query_model_selection.json"      # il loro braccio con il PDT
LORO_MIX = "mix_rtr_query_model_selection.json"


def leggi(percorso: Path):
    """Una riga per gruppo: |phi|, query, configurazione scelta e punteggi."""
    righe = []
    for numero, gruppo in enumerate(json.loads(percorso.read_text(encoding="utf-8"))):
        punteggi = np.asarray(gruppo["results"], dtype=float)
        scelta = gruppo["configs"][int(gruppo["idx_best"])]
        righe.append({
            "phi": int(gruppo["qxm"]),
            "gruppo": numero,
            "query": tuple(int(q) for q in gruppo["query"]),
            "ndcg_scelta": float(punteggi.max()),
            "ndcg_peggiore": float(punteggi.min()),
            "ndcg_media": float(punteggi.mean()),
            "configurazione": scelta,
            **{k: scelta.get(k) for k in IPERPARAMETRI},
        })
    return pd.DataFrame(righe)


def riassunto(t: pd.DataFrame, dataset: str, variante: str):
    r = (t.groupby("phi")
         .agg(gruppi=("gruppo", "count"),
              ndcg_scelta=("ndcg_scelta", "mean"),
              guadagno_sulla_media=("ndcg_scelta", "mean"),
              quanto_e_servito_cercare=("ndcg_scelta", "mean"))
         .reset_index())
    # quanto e' servito cercare: differenza fra la migliore e la media delle configurazioni
    medie = t.groupby("phi")[["ndcg_scelta", "ndcg_media", "ndcg_peggiore"]].mean()
    r["guadagno_sulla_media"] = (medie["ndcg_scelta"] - medie["ndcg_media"]).values
    r["quanto_e_servito_cercare"] = (medie["ndcg_scelta"] - medie["ndcg_peggiore"]).values
    r.insert(0, "variante", variante)
    r.insert(0, "dataset", dataset)
    return r.round(4)


def valori_scelti(t: pd.DataFrame):
    return {k: dict(Counter(t[k]).most_common(3)) for k in IPERPARAMETRI}


def controllo_incrociato(nostro: pd.DataFrame, percorso_loro: Path):
    """Le configurazioni scelte dalla nostra procedura contro quelle dei loro file.

    Il confronto si fa per gruppo di query, identificando i gruppi dalle query che
    contengono e non dalla posizione, perché l'ordine può differire.
    """
    loro = {}
    for gruppo in json.loads(percorso_loro.read_text(encoding="utf-8")):
        punteggi = np.asarray(gruppo["results"], dtype=float)
        configurazioni = gruppo["configs"]
        ammesse = [i for i, c in enumerate(configurazioni)
                   if not any(v for k, v in c.items() if k not in set(IPERPARAMETRI) | {
                       "feature_concat", "feature_sq_diff", "subsample", "sdt_max_leaf_nodes",
                       "min_samples_split", "verbose", "n_jobs_leaf"})]
        migliore = max(ammesse, key=lambda i: punteggi[i])
        loro[tuple(int(q) for q in gruppo["query"])] = {
            "config": {k: configurazioni[migliore].get(k) for k in IPERPARAMETRI},
            "ndcg": float(punteggi[migliore]),
        }

    righe = []
    for _, riga in nostro.iterrows():
        chiave = riga["query"]
        if chiave not in loro:
            continue
        nostra = {k: riga[k] for k in IPERPARAMETRI}
        righe.append({
            "phi": riga["phi"], "gruppo": riga["gruppo"],
            "uguale": nostra == loro[chiave]["config"],
            "ndcg_nostro": round(riga["ndcg_scelta"], 4),
            "ndcg_loro": round(loro[chiave]["ndcg"], 4),
            "scarto": round(riga["ndcg_scelta"] - loro[chiave]["ndcg"], 4),
        })
    return pd.DataFrame(righe)


def main():
    argomenti = [a for a in sys.argv[1:] if not a.startswith("--")]
    base = Path(argomenti[0]) if argomenti else Path("model_selection")
    loro_base = None
    if "--loro" in sys.argv:
        loro_base = Path(sys.argv[sys.argv.index("--loro") + 1])

    # il nome dei file prodotti si puo' cambiare, altrimenti una seconda selezione
    # (per esempio con un numero di fold diverso) sovrascriverebbe la prima, che e'
    # quella con cui sono gia' stati fatti i run. Si chiama nome_uscita e non nome
    # perche' nel ciclo qui sotto `nome` e' gia' il nome del file di model selection:
    # usare lo stesso identificatore lo faceva sovrascrivere dal ciclo
    nome_uscita = sys.argv[sys.argv.index("--nome") + 1] if "--nome" in sys.argv else None

    riassunti, per_run, tabelle = [], {}, {}
    for (dataset, variante), nome in NOSTRI_FILE.items():
        percorso = base / nome
        if not percorso.exists():
            print("manca", percorso)
            continue
        t = leggi(percorso)
        tabelle[(dataset, variante)] = t
        riassunti.append(riassunto(t, dataset, variante))
        per_run.setdefault(dataset, {})[variante] = [
            {"phi": int(r["phi"]), "gruppo": int(r["gruppo"]), "query": list(r["query"]),
             "ndcg_validazione": round(r["ndcg_scelta"], 4), "configurazione": r["configurazione"]}
            for _, r in t.iterrows()]

    if not riassunti:
        return
    print("=== cosa ha scelto la nostra model selection ===")
    print(pd.concat(riassunti, ignore_index=True).to_string(index=False))

    print("\n=== valori ricorrenti ===")
    for chiave, t in tabelle.items():
        print(" ", chiave[0], chiave[1], valori_scelti(t))

    qui = Path(__file__).resolve()
    radice = qui.with_name(nome_uscita) if nome_uscita else qui
    csv = radice.with_suffix(".csv")
    pd.concat(riassunti, ignore_index=True).to_csv(csv, index=False)
    js = radice.with_suffix(".json")
    js.write_text(json.dumps(per_run), encoding="utf-8")
    print("\nscritti", csv.name, "e", js.name)

    if loro_base is not None and (loro_base / LORO_FILE).exists():
        print("\n=== controllo incrociato sul braccio con il PDT ===")
        for dataset in ["FINDHR"]:
            if (dataset, "rtr") not in tabelle:
                continue
            c = controllo_incrociato(tabelle[(dataset, "rtr")], loro_base / LORO_FILE)
            if c.empty:
                print("  nessun gruppo in comune")
                continue
            uguali = int(c["uguale"].sum())
            print(f"  {dataset}: {uguali} configurazioni identiche su {len(c)} gruppi confrontabili")
            print(f"  scarto medio di punteggio: {c['scarto'].mean():+.4f}, "
                  f"massimo {c['scarto'].abs().max():.4f}")
            c.to_csv(qui.with_name("controllo_incrociato.csv"), index=False)
            print("  scritto controllo_incrociato.csv")


if __name__ == "__main__":
    main()
