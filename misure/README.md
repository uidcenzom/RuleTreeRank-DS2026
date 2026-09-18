# Misure

Script di analisi e tabelle della tesi RTRwRuleCard, raccolti per argomento.
Gli script leggono i risultati salvati da `scripts/esperimento.py` e non
ricalcolano niente: ognuno accetta come argomento la cartella da cui leggere,
così le stesse tabelle si possono rifare su risultati diversi. Ogni CSV sta
accanto allo script che lo produce, e la data di ogni misura è nell'intestazione
dello script.

## `confronto_modelli/`

Le tabelle che mettono i modelli uno accanto all'altro.

- `scala_dei_modelli.py` e il suo CSV: una tabella sola con ranking casuale, solo
  primo stadio, kNN euclideo dentro la foglia, RTR con il PDT, RTRwRuleCard con
  la GAM, kNN, LambdaMART e i tre modelli non interpretabili usati come limite
  superiore. Due dataset, cinque valori di |phi|, cinque seed per casella
- `confronto_multiseed.py` e i due CSV: RTR con il PDT contro RTRwRuleCard, con
  media e deviazione standard fra i seed e test di Wilcoxon appaiato sulle query.
  Il file `_server` è la versione sui run del server, l'altro quelli del pc
- `distanza_o_aggregazione.py` e il suo CSV: a partire dallo stesso modello di
  riferimento, quanto rende cambiare il modello di distanza e quanto rende
  cambiare l'aggregazione dei vicini, con lo stesso test appaiato
- `confronto_pdt_gam_findhrlist.py`: il primo confronto fra le due distanze su
  FINDHRlist, a un solo seed

## `iperparametri/`

Cosa conta fra gli iperparametri e come si scelgono.

- `peso_degli_iperparametri.py` e i tre CSV: effetto medio di ogni valore,
  quanto costa fissare un iperparametro invece di sceglierlo gruppo per gruppo,
  e quanto vale scegliere la configurazione per gruppo invece di una sola per
  tutti. Ricavati dai file di model selection del gruppo, senza addestrare nulla
- `configurazioni_per_gruppo.py`, il suo CSV e il JSON: la configurazione
  migliore per ogni gruppo di query, pronta da passare ai run

## `analisi/`

Domande singole, ognuna con la sua risposta misurata.

- `perche_findhr_guadagna_di_piu.py`: perché il vantaggio della GAM è più grande
  sul dataset dove la distanza incide su meno documenti
- `fedelta.py`: verifica che PDT e GAM ricevano esattamente lo stesso bersaglio
- `stat_rulecard.py`: statistiche della GAM, round effettivi, colonne usate,
  quante volte si ripiega sulla distanza euclidea
- `rtr_varianti.py`: prime varianti di RTR provate il 15 settembre

## `log/`

Log delle esecuzioni lunghe, tenuti per poter risalire a quando e come sono
stati prodotti i risultati.

## Dove sono i risultati

In `results/` e `results_server/`, una cartella per esecuzione nella forma
`dataset/modello/phi<n>/seed<s>/`, con la configurazione completa, le metriche e
l'NDCG di ogni query. Modelli e predizioni restano in locale perché pesano.
