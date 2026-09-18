# Misure

Script di analisi e tabelle prodotte durante la tesi RTRwRuleCard, raccolti per
data. Gli script leggono i risultati dei run salvati da `scripts/esperimento.py`
e non ricalcolano niente: ognuno accetta come argomento la cartella da cui
leggere, così si possono rifare le tabelle su risultati diversi.

- `2026-09-15/`: prime misure su MQ2007 e FINDHR, statistiche della GAM e
  fedeltà del bersaglio rispetto al PDT
- `2026-09-16/`, `2026-09-17/`: confronto fra RTR con PDT e RTRwRuleCard su più
  seed, con media, deviazione standard e test di Wilcoxon appaiato sulle query
- `2026-09-18/`:
  - `scala_dei_modelli.py` e il suo CSV: una tabella sola con ranking casuale,
    solo primo stadio, kNN euclideo in foglia, RTR, RTRwRuleCard, kNN,
    LambdaMART e i tre upper bound, su due dataset e cinque |phi|, cinque seed
  - `peso_degli_iperparametri.py` e i tre CSV: quanto pesa ogni iperparametro e
    quanto vale scegliere la configurazione gruppo per gruppo, ricavati dai file
    di model selection del gruppo senza addestrare nulla
  - `configurazioni_per_gruppo.py`: la configurazione migliore per ogni gruppo di
    query, ricavata dai file di model selection
  - `perche_findhr_guadagna_di_piu.py`: perché il vantaggio della GAM è più
    grande sul dataset dove la distanza incide su meno documenti

I risultati dei run stanno in `results/` e `results_server/`: sono versionati
solo i file leggeri (configurazione, metriche, NDCG per query), mentre modelli e
predizioni restano in locale perché pesano.
