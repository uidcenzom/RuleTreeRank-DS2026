"""
Raccoglie in una sola cartella i file leggeri dei run, da pubblicare nel repository.

Le cartelle in cui le macchine scrivono (`results`, `results_server`,
`risultati_scelta`) restano fuori dal controllo di versione: sono spazi di lavoro,
e il fatto che il repository ne tracciasse due ha gia' causato un guaio, perche'
sul server la cartella di lavoro coincideva con quella pubblicata e i numeri del
pc sono finiti dentro quella del server.

Qui invece la pubblicazione e' un atto esplicito. Di ogni run si copiano solo
configurazione, metriche e NDCG per query; modelli e predizioni restano locali
perche' pesano. La struttura resta `dataset/variante/phi<n>/seed<s>/`, quindi gli
script di analisi funzionano senza modifiche.

Quando lo stesso run esiste in piu' cartelle, i punteggi vengono confrontati: se
coincidono si pubblica la copia della sorgente indicata per prima, se non
coincidono il run non viene pubblicato e viene segnalato, perche' due copie dello
stesso esperimento che danno numeri diversi sono un problema da capire, non da
risolvere scegliendo a caso.

Il motivo tipico, verificato il 19 settembre, e' che le due copie vengono da
commit diversi e da versioni diverse delle librerie: fra numpy 2.4.6 sul pc e
2.5.3 sul server cambiano la sequenza dei numeri casuali e l'ordine fra vicini
equidistanti, quindi il ranking casuale e il kNN danno numeri leggermente
diversi. In quel caso la copia buona e' quella della macchina in uso, ma la
scelta va fatta apposta: serve `--risolvi-con-la-prima`, che pubblica la copia
della prima sorgente elencando i run risolti.

Uso:
  python pubblica_risultati.py [destinazione] [sorgente ...] [--risolvi-con-la-prima]
"""
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

LEGGERI = ("config.json", "metriche.json", "ndcg_per_query.csv")
TOLLERANZA = 1e-6


def punteggi(cartella: Path):
    """I valori di NDCG salvati in un run, per confrontare due copie."""
    f = cartella / "metriche.json"
    if not f.exists():
        return None
    m = json.loads(f.read_text(encoding="utf-8"))
    return {k: v["media"] for k, v in m.items() if k.startswith("ndcg@") and isinstance(v, dict)}


def raccogli(sorgenti):
    """Tutti i run trovati: chiave -> lista di cartelle, nell'ordine delle sorgenti."""
    trovati = defaultdict(list)
    for s in sorgenti:
        base = Path(s)
        if not base.exists():
            print(f"  sorgente assente, la salto: {s}")
            continue
        for f in base.rglob("metriche.json"):
            chiave = "/".join(f.relative_to(base).parts[:-1])
            trovati[chiave].append(f.parent)
    return trovati


def differenza(a: dict, b: dict):
    """Massimo scarto fra due insiemi di punteggi, o None se non confrontabili."""
    comuni = set(a) & set(b)
    if not comuni:
        return None
    return max(abs(a[k] - b[k]) for k in comuni)


def main():
    risolvi = "--risolvi-con-la-prima" in sys.argv
    argomenti = [a for a in sys.argv[1:] if not a.startswith("--")]
    destinazione = Path(argomenti[0]) if argomenti else Path("risultati")
    sorgenti = argomenti[1:] or ["risultati_scelta", "results_server", "results"]

    print(f"destinazione: {destinazione}")
    print(f"sorgenti, in ordine di precedenza: {', '.join(sorgenti)}\n")

    trovati = raccogli(sorgenti)
    pubblicati = doppi_concordi = 0
    conflitti, risolti, byte = [], [], 0

    for chiave, cartelle in sorted(trovati.items()):
        if len(cartelle) > 1:
            valori = [punteggi(c) for c in cartelle]
            scarto = max((differenza(valori[0], v) or 0.0) for v in valori[1:])
            if scarto > TOLLERANZA:
                if not risolvi:
                    conflitti.append((chiave, scarto, [str(c) for c in cartelle]))
                    continue
                risolti.append((chiave, scarto, str(cartelle[0])))
            else:
                doppi_concordi += 1

        scelta = cartelle[0]
        arrivo = destinazione / chiave
        arrivo.mkdir(parents=True, exist_ok=True)
        for nome in LEGGERI:
            f = scelta / nome
            if f.exists():
                shutil.copy2(f, arrivo / nome)
                byte += f.stat().st_size
        pubblicati += 1

    print(f"run pubblicati: {pubblicati} ({byte / 1024 / 1024:.1f} MB)")
    print(f"run trovati in piu' cartelle con punteggi identici: {doppi_concordi}")
    if risolti:
        print(f"\ncopie diverse risolte tenendo la prima sorgente: {len(risolti)}")
        print(f"  scarto massimo fra le copie: {max(s for _, s, _ in risolti):.6f}")
        for chiave, scarto, scelta in risolti[:5]:
            print(f"  {chiave}: scarto {scarto:.6f}, pubblicata {scelta}")
        if len(risolti) > 5:
            print(f"  e altri {len(risolti) - 5}")
    if conflitti:
        print(f"\nNON pubblicati, due copie con punteggi diversi: {len(conflitti)}")
        for chiave, scarto, dove in conflitti[:10]:
            print(f"  {chiave}: scarto massimo {scarto:.6f} fra {', '.join(dove)}")
    else:
        print("nessun conflitto fra copie diverse dello stesso run")


if __name__ == "__main__":
    main()
