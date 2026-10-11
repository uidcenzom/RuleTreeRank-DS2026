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
- `configurazione_scelta_fold10.py`: lo stesso confronto con la selezione a dieci
  fold su entrambi i bracci, che è quella che sta in tesi. Produce
  `configurazione_scelta_fold10.csv`. È la misura da cui viene il risultato che a
  configurazione scelta i due bracci sono indistinguibili (+0.0007, nessuna
  combinazione significativa)
- `sottocampionamento.py`: cosa succede riducendo la frazione di documenti della
  cella da cui si costruiscono le coppie, con la colonna `quota_con_gam` che dice
  quante celle costruiscono ancora la GAM invece di ripiegare sull'euclidea.
  Produce `sottocampionamento.csv`
- `costi.csv`: i rapporti di costo fra i due modelli di distanza, letti dai run e
  non riscritti a mano nei grafici
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
- `effetto_della_selezione.py`: quanto rende la selezione per gruppo su ciascuno
  dei due bracci, cioè i due regimi messi uno accanto all'altro. Dice di chi sia
  il merito del cambiamento: la selezione rende +0.0020 al PDT e -0.0031 alla GAM
  su FINDHR, +0.0040 e +0.0029 su FINDHRlist
- `profondita_senza_effetto.py`: perché sul braccio additivo `pdt_depth` non
  cambia il punteggio di convalida in nessuno dei 202 gruppi, mentre sul braccio
  con il PDT lo cambia in tutti. Fa i tre controlli in ordine, compreso quello
  che ha scartato la prima spiegazione che avevo scritto. Va eseguito sul server,
  dove stanno i file della selezione
- `quanto_divergono_i_bracci.py`: quanto si allontanano le configurazioni scelte
  per i due bracci. Serve a sapere che cosa confronta davvero il risultato a
  configurazione scelta: la configurazione coincide in 9 gruppi su 202 su FINDHR
  e in 4 su FINDHRlist, quindi si confrontano due sistemi regolati e non due
  distanze dentro la stessa pipeline
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
- `rumore_del_seme.py`: sotto quale differenza di NDCG@10 due modelli non si
  distinguono, misurato invece che stabilito a occhio. Dà la deviazione standard
  di 0.0010 e i percentili usati come soglia nel capitolo sperimentale
- `granularita.py` e `granularita_sintesi.py`: quanti valori di distanza distinti
  producono i due modelli sulle celle vere, con la configurazione dei run e non
  con una ridotta. Il PDT non supera i sedici valori, la GAM arriva a 132
- `interpretabilita.py`: quanto è lunga e quanto è stabile una spiegazione del
  modello additivo, cioè quante regole servono per il 90% del contributo assoluto
  e se due semi scelgono le stesse colonne

## `figure_tesi/`

Le figure della tesi che vanno costruite da un modello allenato e non da una
tabella: `architettura.py` disegna i due stadi su un esempio vero e
`esempio_spiegazione.py` la spiegazione di una singola coppia. I nomi delle
feature arrivano dal dataset e non sono scritti a mano.

`grafici_tesi.py`, nella cartella principale, produce gli altri grafici dai CSV.

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
