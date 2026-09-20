"""
La stessa selezione fatta con tre fold e con dieci: cosa cambia.

Con tre fold ogni addestramento della convalida usa due terzi dei documenti del
gruppo, con dieci ne usa il 90%. Se l'ipotesi e' giusta, cioe' che con meno dati
le celle sono piu' piccole e viene scelto un primo stadio piu' basso di quello
giusto, allora passando a dieci fold le profondita' scelte devono salire e il
danno sul test deve ridursi.

Lo script confronta tre cose:

1. quali profondita' vengono scelte nei due casi, gruppo per gruppo;
2. quanto spesso la configurazione scelta cambia;
3. se i run riaddestrati esistono, come vanno sul test le due scelte, con il test
   di Wilcoxon appaiato sulle query.

Il punto 3 e' l'unico che conta davvero: i primi due dicono che la selezione si e'
spostata, il terzo dice se si e' spostata nella direzione giusta.

I run riaddestrati con le due selezioni si chiamano allo stesso modo, `rtr_scelta`,
e si distinguono solo per la cartella in cui stanno: vanno quindi tenuti separati.

Uso:
  python tre_fold_contro_dieci.py [selezione 3 fold] [selezione 10 fold] [run 10 fold] [run 3 fold e fissi ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

IPER = ["feature_diff", "pdt_depth", "n_neighbors", "sdt_depth", "dist_objective"]
DATASET = ["FINDHR", "FINDHRLIST"]
PHI = [1, 2, 4, 6, 10]


def scelte(percorso: Path):
    """Per ogni gruppo: |phi|, query e configurazione scelta."""
    if not percorso.exists():
        return None
    righe = []
    for numero, gruppo in enumerate(json.loads(percorso.read_text(encoding="utf-8"))):
        conf = gruppo["configs"][int(gruppo["idx_best"])]
        righe.append({"phi": int(gruppo["qxm"]), "gruppo": numero,
                      "query": tuple(int(q) for q in gruppo["query"]),
                      "ndcg_validazione": float(max(gruppo["results"])),
                      **{k: conf.get(k) for k in IPER}})
    return pd.DataFrame(righe)


def per_query(cartelle, dataset, variante, phi):
    serie, visti = [], set()
    for base in cartelle:
        for d in sorted(Path(base).glob(f"{dataset}/{variante}/phi{phi}/seed*")):
            f = d / "ndcg_per_query.csv"
            if d.name in visti or not f.exists():
                continue
            visti.add(d.name)
            serie.append(pd.read_csv(f).set_index("query")["completo"])
    return pd.concat(serie, axis=1).mean(axis=1) if serie else None


def confronta_scelte(tre: pd.DataFrame, dieci: pd.DataFrame, dataset: str):
    # i gruppi si accoppiano dalle query che contengono, non dalla posizione: le due
    # selezioni possono coprire un insieme diverso di |phi|, e allora l'indice dentro
    # il file slitta e l'accoppiamento per posizione perderebbe dei gruppi in silenzio
    unito = tre.merge(dieci, on=["phi", "query"], suffixes=("_3", "_10"))
    comuni = set(tre["phi"]) & set(dieci["phi"])
    attesi = len(tre[tre["phi"].isin(comuni)])
    if len(unito) != attesi:
        print(f"  attenzione: {attesi - len(unito)} gruppi su {attesi} non si accoppiano")
    righe = []
    for phi, sotto in unito.groupby("phi"):
        diverse = sum(any(sotto[f"{k}_3"].iloc[i] != sotto[f"{k}_10"].iloc[i] for k in IPER)
                      for i in range(len(sotto)))
        righe.append({
            "dataset": dataset, "phi": phi, "gruppi": len(sotto),
            "profondita_media_3fold": round(sotto["sdt_depth_3"].mean(), 2),
            "profondita_media_10fold": round(sotto["sdt_depth_10"].mean(), 2),
            "sale": int((sotto["sdt_depth_10"] > sotto["sdt_depth_3"]).sum()),
            "scende": int((sotto["sdt_depth_10"] < sotto["sdt_depth_3"]).sum()),
            "config_diversa": diverse,
        })
    return pd.DataFrame(righe)


def main():
    argomenti = sys.argv[1:]
    tre_fold = Path(argomenti[0]) if argomenti else Path("model_selection")
    dieci_fold = Path(argomenti[1]) if len(argomenti) > 1 else Path("model_selection_fold10")
    # i run riaddestrati con le due selezioni hanno lo stesso nome di variante,
    # `rtr_scelta`, e si distinguono solo per la cartella da cui vengono: vanno
    # quindi letti da radici separate, altrimenti gli uni coprirebbero gli altri
    run_dieci = argomenti[2] if len(argomenti) > 2 else "risultati_scelta_fold10"
    run_altri = argomenti[3:] or ["risultati_scelta", "risultati"]

    tabelle = []
    for dataset in DATASET:
        tre = scelte(tre_fold / f"{dataset}_rtr.json")
        dieci = scelte(dieci_fold / f"{dataset}_rtr.json")
        if tre is None or dieci is None:
            print(f"{dataset}: manca una delle due selezioni, la salto")
            continue
        tabelle.append(confronta_scelte(tre, dieci, dataset))

    if tabelle:
        t = pd.concat(tabelle, ignore_index=True)
        print("=== come si sposta la scelta della profondita' del primo stadio ===")
        print("(la configurazione fissa usa 5; con tre fold la media era piu' bassa)")
        print(t.to_string(index=False))
        salita = (t["profondita_media_10fold"] - t["profondita_media_3fold"]).mean()
        print(f"\nvariazione media della profondita' scelta: {salita:+.2f}")
        t.to_csv(Path(__file__).with_suffix(".csv"), index=False)

    print("\n=== effetto sul test, se i run riaddestrati ci sono ===")
    righe = []
    for dataset in DATASET:
        for phi in PHI:
            fissa = per_query(run_altri, dataset, "rtr", phi)
            tre_q = per_query(run_altri, dataset, "rtr_scelta", phi)
            dieci_q = per_query([run_dieci], dataset, "rtr_scelta", phi)
            if dieci_q is None:
                continue
            riga = {"dataset": dataset, "phi": phi}
            for nome, serie in [("scelta 3 fold", tre_q), ("scelta 10 fold", dieci_q)]:
                riga[nome] = round(float(serie.mean()), 4) if serie is not None else None
            if fissa is not None:
                riga["fissa"] = round(float(fissa.mean()), 4)
                for nome, serie in [("3 fold - fissa", tre_q), ("10 fold - fissa", dieci_q)]:
                    if serie is None:
                        continue
                    d = (serie - fissa).dropna()
                    d = d[d != 0]
                    riga[nome] = round(float(serie.mean() - fissa.mean()), 4)
                    riga[f"p {nome}"] = round(float(wilcoxon(d).pvalue), 4) if len(d) else None
            righe.append(riga)
    if righe:
        print(pd.DataFrame(righe).to_string(index=False))
    else:
        print("  i run con le configurazioni a dieci fold non ci sono ancora")


if __name__ == "__main__":
    main()
