"""
Quanto vale il rumore dovuto al solo seme casuale.

Serve a rispondere a una domanda precisa: sotto quale differenza di NDCG@10 due
modelli non si distinguono? La risposta non si stabilisce a occhio, si misura.

Il seme cambia una cosa sola: come gli alberi rompono i pareggi fra divisioni
ugualmente buone. Tutto il resto (dati, codice, iperparametri) resta identico.
Quindi due esecuzioni dello stesso modello con semi diversi danno due punteggi
che dovrebbero essere uguali, e la loro differenza e' rumore puro.

Lo script calcola due famiglie di numeri.

1. Variabilita' del punteggio complessivo: deviazione standard e scarto fra il
   massimo e il minimo fra i semi, per ogni combinazione di dataset, modello e
   numero di query per modello.

2. Il "gain nullo", che e' la misura che conta davvero per noi. Tutti i nostri
   confronti sono appaiati: per ogni query si fa la differenza di NDCG@10 fra
   due modelli e poi si media. Se applichiamo quello stesso conto a due semi
   dello *stesso* modello, il risultato dovrebbe essere zero. Quanto si discosta
   da zero e' esattamente il rumore che un guadagno vero deve superare per
   essere credibile.

La seconda famiglia e' quella da citare quando si parla di soglia, perche' e'
omogenea ai guadagni che riportiamo nelle tabelle.

Uso:
  python rumore_del_seme.py <cartella dei run> [altra cartella ...]
"""
import itertools
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

PHI = [1, 2, 4, 6, 10]
DATASET = ["FINDHR", "FINDHRLIST"]
MODELLI = ["rtr", "rtrwrulecard"]


def esecuzioni(cartelle, dataset, modello, phi):
    """Le cartelle dei semi di una combinazione, senza duplicati fra piu' radici."""
    trovate, visti = [], set()
    for base in cartelle:
        for d in sorted(Path(base).glob(f"{dataset}/{modello}/phi{phi}/seed*")):
            if d.name in visti or not (d / "metriche.json").exists():
                continue
            visti.add(d.name)
            trovate.append(d)
    return trovate


def punteggio(cartella):
    """NDCG@10 complessivo di una esecuzione."""
    return json.loads((cartella / "metriche.json").read_text(encoding="utf-8"))["ndcg@10_completo"]["media"]


def per_query(cartella):
    """NDCG@10 di ogni singola query di test."""
    return pd.read_csv(cartella / "ndcg_per_query.csv").set_index("query")["completo"]


def main():
    cartelle = sys.argv[1:] or ["risultati"]

    righe, nulli = [], []
    for dataset in DATASET:
        for modello in MODELLI:
            for phi in PHI:
                semi = esecuzioni(cartelle, dataset, modello, phi)
                if len(semi) < 2:
                    continue
                punti = [punteggio(d) for d in semi]
                righe.append({
                    "dataset": dataset, "modello": modello, "phi": phi, "semi": len(semi),
                    "media": round(statistics.mean(punti), 5),
                    "deviazione_standard": round(statistics.stdev(punti), 5),
                    "massimo_meno_minimo": round(max(punti) - min(punti), 5),
                })
                # il gain nullo: lo stesso conto dei confronti veri, ma fra due
                # semi dello stesso modello, dove per costruzione non c'e' effetto
                for a, b in itertools.combinations(semi, 2):
                    nulli.append(float((per_query(b) - per_query(a)).mean()))

    if not righe:
        print("nessuna esecuzione trovata nelle cartelle indicate")
        return

    t = pd.DataFrame(righe)
    print("=== variabilita' del punteggio complessivo fra i semi ===")
    print(t.to_string(index=False))

    assoluti = sorted(abs(x) for x in nulli)
    print("\n=== gain nullo: differenza appaiata fra due semi dello stesso modello ===")
    print(f"coppie di semi confrontate: {len(nulli)}")
    print(f"media del valore assoluto:  {statistics.mean(assoluti):.5f}")
    print(f"mediana:                    {statistics.median(assoluti):.5f}")
    print(f"deviazione standard:        {statistics.stdev(nulli):.5f}")
    print(f"95esimo percentile:         {assoluti[int(0.95 * len(assoluti))]:.5f}")
    print(f"massimo:                    {max(assoluti):.5f}")

    print("\nCome si legge: un guadagno appaiato sotto circa "
          f"{statistics.stdev(nulli):.4f} non si distingue dal solo effetto del seme, "
          f"e serve superare {assoluti[int(0.95 * len(assoluti))]:.4f} "
          "per stare fuori dal 95% dei confronti nulli.")

    uscita = Path(__file__).with_suffix(".csv")
    t.to_csv(uscita, index=False)
    riassunto = Path(__file__).with_name("rumore_del_seme_gain_nullo.csv")
    pd.DataFrame({"gain_nullo": nulli}).to_csv(riassunto, index=False)
    print("\nscritti", uscita.name, "e", riassunto.name)


if __name__ == "__main__":
    main()
