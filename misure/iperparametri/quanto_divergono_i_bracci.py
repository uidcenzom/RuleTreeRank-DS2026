"""Quanto si allontanano i due bracci quando ciascuno sceglie la sua configurazione.

A configurazione fissa i due bracci differiscono solo nel modello di distanza, ed
e' questo a rendere il confronto attribuibile. A configurazione scelta la
selezione sceglie anche la profondita' del primo stadio e il numero di vicini, e
li sceglie separatamente sui due bracci: se quelle scelte divergono, il confronto
non e' piu' fra due distanze dentro la stessa pipeline ma fra due pipeline
intere. Prima di leggere il risultato bisogna sapere quanto divergono.

Uso (sul server, dove stanno i file della selezione):
  python3 quanto_divergono_i_bracci.py
"""
import json
from collections import Counter
from pathlib import Path

SELEZIONE = Path("/home/vmanzo/RuleTreeRank-DS2026/model_selection_fold10")
PHI = [1, 2, 4, 6, 10]
PARAMETRI = ["sdt_depth", "pdt_depth", "n_neighbors", "feature_diff", "dist_objective"]


def scelte(dataset, variante):
    """La configurazione scelta per ogni gruppo, nell'ordine dei gruppi."""
    fuori = []
    for phi in PHI:
        f = SELEZIONE / ("%s_%s_phi%d.json" % (dataset, variante, phi))
        for voce in json.loads(f.read_text()):
            c = voce["configs"][int(voce["idx_best"])]
            fuori.append((phi, tuple(voce["query"]), {k: c[k] for k in PARAMETRI}))
    return fuori


for dataset in ("FINDHR", "FINDHRLIST"):
    pdt, gam = scelte(dataset, "rtr"), scelte(dataset, "rtrwrulecard")
    assert len(pdt) == len(gam), (len(pdt), len(gam))
    uguali = Counter()
    identiche = 0
    primo_stadio = Counter()
    for (phi_a, q_a, a), (phi_b, q_b, b) in zip(pdt, gam):
        assert (phi_a, q_a) == (phi_b, q_b), "gruppi disallineati"
        for k in PARAMETRI:
            uguali[k] += a[k] == b[k]
        identiche += a == b
        primo_stadio[(a["sdt_depth"], b["sdt_depth"])] += 1
    n = len(pdt)
    print("=" * 72)
    print("%s: %d gruppi" % (dataset, n))
    for k in PARAMETRI:
        print("   %-16s stesso valore sui due bracci in %3d gruppi su %d (%.0f%%)"
              % (k, uguali[k], n, 100 * uguali[k] / n))
    print("   configurazione identica in %d gruppi su %d (%.0f%%)" % (identiche, n, 100 * identiche / n))
    media_pdt = sum(a["sdt_depth"] for _, _, a in pdt) / n
    media_gam = sum(b["sdt_depth"] for _, _, b in gam) / n
    print("   profondita' media del primo stadio: PDT %.2f, GAM %.2f (riferimento 5)"
          % (media_pdt, media_gam))
