"""
Model selection per gruppo di query sulla griglia del gruppo.

Per ogni valore di |phi| e per ogni gruppo di query allena tutte le configurazioni
della griglia con una convalida a fold sui documenti di ciascuna query e tiene
quella con l'NDCG@10 medio più alto. È la stessa procedura usata dal gruppo, con
una differenza: il modello di distanza può essere la GAM di RuleCard e non solo
il PDT.

I gruppi di query sono quelli di Iommi (`dati_dal_gruppo/`), così i risultati
stanno accanto ai suoi. La griglia è quella mandata dal gruppo il 27 agosto:
feature_diff, pdt_depth, n_neighbors, sdt_depth e dist_objective, cioè 144
configurazioni. pdt_depth conta anche per la GAM, perché è la profondità
dell'albero che RuleCard usa come termine additivo a ogni round.

Il risultato ha la stessa forma dei file *_query_model_selection.json di Iommi,
una voce per gruppo con tutte le configurazioni, il punteggio di ognuna e
l'indice della migliore:

  <risultati>/<dataset>_<variante>_phi<n>.json    un file per |phi|, per poter
                                                  riprendere dopo un'interruzione
  <risultati>/<dataset>_<variante>.json           tutti i |phi| insieme

Esempio:
  python scripts/model_selection.py --dataset FINDHR --variante rtrwrulecard \
      --phi 1 --processi 128
"""
import argparse
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

from ltr_utility.dataset import load_by_query_dataset, DatasetName
from ltr_utility.model_selection.query_model_selection import train_evaluate_config
from experiments.varianti import WrapperMixRTRVariante

K_NDCG = 10
GRUPPI_IOMMI = REPO / "dati_dal_gruppo" / "query_similarity_iommi_2026-09-18.json"

# griglia del gruppo, 27 agosto
GRIGLIA = {
    "feature_diff": [True, False],
    "pdt_depth": [2, 4, 6],
    "n_neighbors": [3, 4, 5],
    "sdt_depth": [2, 4, 6, 8],
    "dist_objective": ["dist", "y"],
}

# valori tenuti fissi dal gruppo
FISSI = dict(feature_concat=True, feature_sq_diff=False, subsample=1.0, sdt_max_leaf_nodes=None,
             min_samples_split=2, verbose=False, n_jobs_leaf=1)

# parametri della GAM: lr 1 come indicato da Landi, patience e round del paper di RuleCard
RULECARD = dict(distanza="rulecard", rulecard_lr=1.0, rulecard_max_n_iter=100, rulecard_patience=15)


def leggi_argomenti():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["FINDHR", "FINDHRLIST", "MQ"])
    p.add_argument("--variante", required=True, choices=["rtr", "rtrwrulecard"])
    p.add_argument("--phi", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10, 50, 100])
    p.add_argument("--fold", type=int, default=3, help="fold della convalida dentro ogni gruppo")
    p.add_argument("--processi", type=int, default=1, help="configurazioni allenate in parallelo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gruppi", type=Path, default=GRUPPI_IOMMI, help="file query_similarity da usare")
    p.add_argument("--dati", type=Path, default=REPO / "datasets")
    p.add_argument("--risultati", type=Path, default=REPO / "model_selection")
    p.add_argument("--max-gruppi", type=int, default=None, help="solo i primi N gruppi, per le prove")
    p.add_argument("--max-config", type=int, default=None, help="solo le prime N configurazioni, per le prove")
    return p.parse_args()


def configurazioni(args):
    """La griglia, come lista di dizionari pronti per il wrapper."""
    nomi = list(GRIGLIA)
    lista = []
    for valori in product(*(GRIGLIA[n] for n in nomi)):
        conf = dict(zip(nomi, valori), **FISSI, random_state=args.seed)
        if args.variante == "rtrwrulecard":
            conf.update(RULECARD)
        else:
            conf["distanza"] = "pdt"
        lista.append(conf)
    return lista[:args.max_config] if args.max_config else lista


def gruppi_di_query(percorso: Path, phi: int, max_gruppi):
    """I gruppi di query per un |phi|, letti da un file query_similarity."""
    tutti = pd.read_json(percorso)
    if phi not in tutti.columns:
        raise SystemExit(f"|phi| = {phi} non c'è in {percorso.name}: ci sono {list(tutti.columns)}")
    gruppi = [[int(q) for q in g] for g in tutti[phi].dropna().tolist()]
    return gruppi[:max_gruppi] if max_gruppi else gruppi


def valuta_gruppo(train, query, configs, fold, processi):
    """Punteggio di ogni configurazione su un gruppo di query."""
    sotto = train[query]
    dati = (np.asarray(sotto.x), np.asarray(sotto.y), np.asarray(sotto.q))
    lavoro = [delayed(train_evaluate_config)(m=WrapperMixRTRVariante, conf=c, tr=dati,
                                             eval_at=K_NDCG, fold=fold) for c in configs]
    if processi == 1:
        return [train_evaluate_config(m=WrapperMixRTRVariante, conf=c, tr=dati,
                                      eval_at=K_NDCG, fold=fold) for c in configs]
    # ogni configurazione usa un processo solo, il parallelismo sta qui fuori
    with parallel_backend("loky", inner_max_num_threads=1):
        return Parallel(n_jobs=min(processi, len(configs)), batch_size="auto")(lavoro)


def esegui_phi(args, phi, train, configs):
    uscita = args.risultati / f"{args.dataset}_{args.variante}_phi{phi}.json"
    if uscita.exists():
        print(f"|phi|={phi}: già fatto, salto ({uscita.name})", flush=True)
        return json.loads(uscita.read_text(encoding="utf-8"))

    gruppi = gruppi_di_query(args.gruppi, phi, args.max_gruppi)
    print(f"|phi|={phi}: {len(gruppi)} gruppi per {len(configs)} configurazioni, "
          f"{args.fold} fold, inizio {datetime.now():%H:%M:%S}", flush=True)

    voci = []
    for numero, query in enumerate(gruppi):
        t0 = time.time()
        punteggi = [float(p) for p in valuta_gruppo(train, query, configs, args.fold, args.processi)]
        migliore = int(np.argmax(punteggi))
        voci.append({"qxm": phi, "query": query, "configs": configs,
                     "idx_best": migliore, "results": punteggi})
        print(f"  gruppo {numero + 1}/{len(gruppi)}: migliore {punteggi[migliore]:.4f} "
              f"(config {migliore}), {time.time() - t0:.0f}s", flush=True)

    uscita.parent.mkdir(parents=True, exist_ok=True)
    uscita.write_text(json.dumps(voci), encoding="utf-8")
    print(f"|phi|={phi}: scritto {uscita.name}", flush=True)
    return voci


def main():
    args = leggi_argomenti()
    configs = configurazioni(args)
    _, _, _, train = load_by_query_dataset(args.dati, DatasetName[args.dataset],
                                           hold_out=(0.5, 0.2, 0.3), verbose=False)
    print(f"{args.dataset}: {train} | griglia di {len(configs)} configurazioni | "
          f"gruppi da {args.gruppi.name}", flush=True)

    tutto = []
    for phi in args.phi:
        tutto.extend(esegui_phi(args, phi, train, configs))

    insieme = args.risultati / f"{args.dataset}_{args.variante}.json"
    insieme.write_text(json.dumps(tutto), encoding="utf-8")
    print("scritto", insieme, flush=True)


if __name__ == "__main__":
    main()
