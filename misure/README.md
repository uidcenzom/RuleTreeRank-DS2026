# Misure

Script di analisi e tabelle della tesi RTRwRuleCard, raccolti per argomento.
Gli script leggono i risultati salvati da `scripts/esperimento.py` e non
addestrano niente: ognuno accetta come argomento la cartella da cui leggere,
così le stesse tabelle si possono rifare su risultati diversi. Ogni CSV sta
accanto allo script che lo produce.

Chi vuole solo i risultati commentati, senza entrare qui, trova tutto in
`risultati_spiegati/`.

## `confronto_modelli/`

Le tabelle che mettono i modelli uno accanto all'altro.

- `scala_dei_modelli.py`: una tabella sola con ranking casuale, solo primo
  stadio, kNN euclideo dentro la foglia, RTR con il PDT, RTRwRuleCard con la
  GAM, kNN, LambdaMART e i tre modelli non interpretabili usati come limite
  superiore. Due dataset, cinque valori di |phi|, cinque seed per casella
- `confronto_multiseed.py`: RTR con il PDT contro RTRwRuleCard, con media e
  deviazione standard fra i seed, il **gain** appaiato (differenza di NDCG@10
  query per query, poi la media) e il test di Wilcoxon su quelle differenze.
  Produce `confronto_multiseed.csv`; `confronto_multiseed_server.csv` ne è una
  copia con lo stesso contenuto, tenuta perché i documenti la citano con quel
  nome (la vecchia distinzione fra run del pc e run del server è sparita quando
  i risultati sono stati unificati in `risultati/`)
- `distanza_o_aggregazione.py`: a partire dallo stesso riferimento, quanto rende
  cambiare il modello di distanza e quanto rende sostituire la correzione dentro
  la cella con una foresta, con lo stesso test appaiato
- `varianti_del_modello.py`: le varianti del primo e del secondo stadio (kNN
  pesato per distanza, vincolo sui documenti per foglia, foresta nella cella)
  contro RTR con il PDT. Unisce più cartelle di run senza contare due volte lo
  stesso seme
- `ablation.py`: le quattro varianti della figura di ablation del paper
  (completo, senza informazione di query, solo r(x), solo s(x)) per entrambi i
  modelli di distanza
- `configurazione_scelta.py`: il confronto fra i due modelli a configurazione
  fissa e a configurazione scelta per gruppo
- `confronto_pdt_gam_findhrlist.py`: il primo confronto fra le due distanze, a un
  solo seed e solo su FINDHRlist. Superato da `confronto_multiseed.py`, tenuto
  perché riporta anche statistiche della GAM e tempi che gli altri non hanno

## `iperparametri/`

Cosa conta fra gli iperparametri e come si scelgono.

- `peso_degli_iperparametri.py` e i tre CSV: effetto medio di ogni valore, quanto
  costa fissare un iperparametro invece di sceglierlo gruppo per gruppo, e quanto
  vale scegliere la configurazione per gruppo invece di una sola per tutti.
  Ricavati dai file di model selection del gruppo, senza addestrare nulla
- `configurazioni_per_gruppo.py`: la configurazione migliore per ogni gruppo di
  query ricavata dai file del gruppo, pronta da passare ai run
- `configurazioni_scelte.py`: le configurazioni scelte dalla nostra model
  selection, con il controllo incrociato contro quelle del gruppo
- `tre_fold_contro_dieci.py`: la stessa selezione fatta con tre fold e con dieci.
  Confronta le profondità scelte nei due casi e, se i run riaddestrati esistono,
  l'effetto sul test. È la misura che mostra che il difetto sta nella procedura
  di selezione e si corregge aumentando i fold

## `analisi/`

Domande singole, ognuna con la sua risposta misurata e il suo file di output.

- `fedelta.py`: quanto la distanza appresa somiglia all'euclidea al quadrato
  (Spearman sulle coppie), quanti vicini in comune con l'euclidea, e quanti
  valori distinti produce. È la misura dietro due spiegazioni che usiamo spesso:
  perché la profondità dell'albero di distanza non sposta i punteggi, e perché
  pesare i vicini per la distanza non aiuta
- `stat_rulecard.py`: statistiche della GAM, round effettivi, quante celle
  costruiscono il modello e quante ripiegano sulla distanza euclidea, con i
  motivi del ripiego
- `perche_findhr_guadagna_di_piu.py`: perché il vantaggio della GAM è più grande
  sul dataset dove la distanza incide su meno documenti

## `log/`

Log delle esecuzioni lunghe, tenuti per poter risalire a quando e come sono
stati prodotti i risultati.

## Dove sono i risultati

In `risultati/`, una cartella per esecuzione nella forma
`dataset/modello/phi<n>/seed<s>/`, con la configurazione completa, le metriche e
l'NDCG di ogni query di test. Modelli e predizioni restano in locale perché
pesano. La pubblicazione si fa con `pubblica_risultati.py`, che copia i file
leggeri dalle cartelle di lavoro e si rifiuta di scegliere fra due copie in
disaccordo.
