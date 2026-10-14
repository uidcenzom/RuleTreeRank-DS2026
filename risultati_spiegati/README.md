# Risultati spiegati

Questa cartella è una guida ai risultati per chi non lavora dentro il repository.
Contiene solo due cose: le tabelle che servono e il commento che le rende
leggibili. Chi vuole anche vedere **come** sono calcolati i numeri trova accanto
a questo file il notebook `risultati.ipynb`: ricalcola tutto dai file delle
esecuzioni, gira in pochi secondi e non richiede i dataset. Ogni tabella è un CSV in questa stessa cartella, e per ognuna è
indicato da quale script di `misure/` proviene, se qualcuno volesse risalire al
calcolo.

Aggiornata all'11 ottobre 2026.

---

## 1. Come è fatto il modello, e quali sono i due pezzi che tocchiamo

RuleTreeRank lavora in due stadi.

**Primo stadio.** Un albero di regole poco profondo divide i documenti e assegna
a ognuno un punteggio di base `r(x)`. Le foglie dell'albero, incrociate con la
query, definiscono delle **celle**: una cella è l'insieme dei documenti di una
certa query che cadono in una certa foglia.

**Secondo stadio.** Dentro ogni cella il punteggio viene corretto guardando i
documenti vicini. Questo stadio è fatto di **due pezzi distinti**, ed è
importante tenerli separati perché i nostri due esperimenti principali ne toccano
uno ciascuno:

| pezzo | cosa fa | cosa ci abbiamo messo |
|---|---|---|
| **distanza appresa** | decide *quali* documenti della cella sono vicini fra loro | PDT (albero di distanza a coppie, l'originale) oppure la **GAM di RuleCard** (modello additivo) |
| **aggregatore** | decide *come* si combinano i vicini per correggere il punteggio | kNN: prende i k vicini scelti al passo precedente e ne fa la media delle etichette (l'originale) |

Quando scriviamo «abbiamo cambiato la distanza» intendiamo il primo pezzo: il
PDT sostituito dalla GAM di RuleCard, con il kNN che resta identico.

**La foresta dentro la cella è invece un'altra cosa, e va detta con precisione.**
Non è una funzione di aggregazione che prende k vicini e ne restituisce uno:
sostituisce **tutto il secondo stadio dentro la cella**. Non usa la distanza
appresa e non recupera nessun vicino. Addestra una `RandomForestRegressor` sui
documenti della cella, dalle feature al residuo, e in predizione risponde
direttamente la foresta. Nel codice è `ForestaInFoglia` in
`experiments/varianti.py`: eredita l'interfaccia di `KNNRegFast` solo perché RTR
costruisce l'aggregatore con quei parametri e poi gli chiama `fit` e `predict`,
ma `n_neighbors` non entra nella predizione.

## 2. Glossario minimo

- **seed**: il seme del generatore casuale. Gli alberi rompono a caso i pareggi
  fra divisioni ugualmente buone, quindi lo stesso codice sugli stessi dati dà
  risultati un po' diversi a ogni esecuzione. Ogni configurazione è perciò
  eseguita **cinque volte con cinque semi diversi** e i numeri riportati sono le
  medie. Quanto pesa il seme è misurato, non stimato
  (`misure/analisi/rumore_del_seme.py`): applicando lo stesso confronto appaiato a
  due semi dello stesso modello, il guadagno apparente ha deviazione standard 0.0010
  e nel 95% dei casi sta sotto 0.0023. È il metro con cui giudicare se una
  differenza è reale.
- **i due dataset**: **FINDHR** e **FINDHRℓ** (la variante con l'etichetta
  listwise). Ogni query ha esattamente 280 documenti.
- **|φ|**: quante query vede un singolo modello. Usiamo |φ| ∈ {1, 2, 4, 6, 10}.
  «Dieci combinazioni» significa 2 dataset × 5 valori di |φ|.
- **NDCG@10**: la metrica di qualità dell'ordinamento, fra 0 e 1. Tutte le
  differenze riportate sono differenze di NDCG@10.
- **p**: il p-value del test di Wilcoxon appaiato sulle query. Sotto 0.05 diciamo
  che la differenza è distinguibile dal rumore.

## 3. Risultato principale: la GAM batte il PDT, a parità di tutto il resto

Stessa architettura, stessi iperparametri, stessi semi: cambia **solo** il
modello di distanza.

| dataset | \|φ\| | PDT | GAM | differenza | p |
|---|---|---|---|---|---|
| FINDHR | 1 | 0.9521 | 0.9535 | +0.0014 | 0.772 |
| FINDHR | 2 | 0.9545 | 0.9602 | +0.0057 | **0.014** |
| FINDHR | 4 | 0.9539 | 0.9602 | +0.0063 | 0.103 |
| FINDHR | 6 | 0.9567 | 0.9648 | +0.0080 | **0.002** |
| FINDHR | 10 | 0.9516 | 0.9593 | +0.0077 | **0.002** |
| FINDHRℓ | 1 | 0.9770 | 0.9784 | +0.0014 | **0.016** |
| FINDHRℓ | 2 | 0.9752 | 0.9764 | +0.0012 | 0.349 |
| FINDHRℓ | 4 | 0.9750 | 0.9764 | +0.0014 | **0.025** |
| FINDHRℓ | 6 | 0.9728 | 0.9748 | +0.0020 | 0.073 |
| FINDHRℓ | 10 | 0.9741 | 0.9769 | +0.0027 | **0.002** |

**Media +0.0038, positiva in 10 combinazioni su 10, significativa in 6 su 10.**
Vale a parità di configurazione: con la configurazione scelta per gruppo su
entrambi i bracci il margine scende a +0.0007 e non è significativo in nessuna
combinazione, vedi la sezione 8.

File: `confronto_multiseed_server.csv`. La colonna **`gain`** è il confronto
appaiato: per ogni query la differenza di NDCG@10 fra i due modelli, poi la media
di quelle differenze. Coincide con la colonna `differenza` perché l'NDCG
complessivo che salviamo è la media semplice di quelli per query, quindi fare
prima le differenze o prima le medie dà lo stesso numero.

## 4. Il collo di bottiglia non è la distanza, è la correzione dentro la cella

Stesso riferimento (RTR con PDT e kNN), due modifiche alternative: cambiare la
distanza appresa lasciando il kNN, oppure sostituire tutta la correzione dentro
la cella con la foresta.

| dataset | \|φ\| | guadagno cambiando la **distanza** | guadagno sostituendo la **correzione nella cella** |
|---|---|---|---|
| FINDHR | 1 | +0.0014 | +0.0055 |
| FINDHR | 2 | +0.0057 | +0.0093 |
| FINDHR | 4 | +0.0063 | +0.0124 |
| FINDHR | 6 | +0.0080 | +0.0132 |
| FINDHR | 10 | +0.0077 | +0.0136 |
| FINDHRℓ | 1 | +0.0014 | +0.0036 |
| FINDHRℓ | 2 | +0.0012 | +0.0054 |
| FINDHRℓ | 4 | +0.0014 | +0.0066 |
| FINDHRℓ | 6 | +0.0020 | +0.0084 |
| FINDHRℓ | 10 | +0.0027 | +0.0085 |
| **media** | | **+0.0038** | **+0.0086** |

> **Attenzione a come si legge.** La foresta dentro la cella **non è
> interpretabile**: guarda direttamente le feature dei documenti e smette di
> spiegare perché un documento viene prima di un altro. Non è un modello che
> proponiamo, è un **upper bound**: serve a misurare quanto si lascia sul tavolo
> restando leggibili. Il risultato non dice «usate la foresta», dice «se si vuole
> guadagnare, il pezzo su cui conviene lavorare è quello che succede dentro la
> cella, non la distanza che sceglie i vicini».

File: `distanza_o_aggregazione.csv`. Anche qui `gain_distanza` e
`gain_aggregazione` sono gli stessi guadagni calcolati come media delle
differenze query per query.

Rimisurato con cinque semi, quel +0.0086 è positivo e significativo in **dieci
combinazioni su dieci**: è l'evidenza più solida che abbiamo, più forte del
risultato principale fra GAM e PDT, che è 10 su 10 positivo ma 6 su 10
significativo.

### Cosa abbiamo provato che non funziona

Tutte a cinque semi, sulle stesse dieci combinazioni e contro lo stesso
riferimento:

| variante | media | positiva | significativa |
|---|---|---|---|
| kNN pesato per 1/distanza invece che uniforme | **−0.0060** | 0 su 10 | 8 su 10 |
| minimo 6 documenti di training per foglia | −0.0009 | 4 su 10 | 2 su 10 |
| minimo 10 documenti di training per foglia | −0.0035 | 2 su 10 | 2 su 10 |

Il kNN pesato non è un pareggio: peggiora in tutte e dieci le combinazioni, in
otto in modo significativo. È coerente con il resto del quadro — nelle celle
piccole k è quasi la dimensione della cella, quindi il kNN prende comunque quasi
tutti i documenti e pesare i valori della distanza aggiunge solo varianza. Il
vincolo sulla dimensione delle foglie non sposta niente.

Messo insieme al punto qui sopra: **finora nessuna modifica interpretabile
migliora il secondo stadio, e l'unica cosa che guadagna non è interpretabile.**
Lo spazio aperto è un aggregatore che guadagni restando leggibile, ed è quello
su cui stiamo lavorando adesso.

File: `varianti_del_modello.csv`.

## 5. Dove siamo, fra il caso e i modelli non interpretabili

Intervalli sulle dieci combinazioni (NDCG@10):

| modello | interpretabile | intervallo |
|---|---|---|
| ordinamento casuale | — | 0.51 – 0.67 |
| solo il primo stadio `r(x)` | sì | 0.949 – 0.973 |
| kNN con distanza euclidea nella foglia | sì | 0.953 – 0.979 |
| **RTR (PDT)** | sì | 0.952 – 0.977 |
| **RTRwRuleCard (GAM)** | sì | 0.954 – 0.978 |
| foresta dentro la cella | no | 0.958 – 0.983 |
| foresta sul gruppo di query | no | 0.974 – 0.982 |
| boosting sul gruppo di query | no | 0.963 – 0.984 |
| LambdaMART | no | 0.918 – 0.985 |

La distanza dal soffitto non interpretabile è in media **0.0128** di NDCG@10:
copriamo fra il 92% e il 99% della scala che separa il caso dal miglior modello
non interpretabile. È anche il motivo per cui le differenze fra PDT e GAM sono
piccole in assoluto: c'è poco spazio sopra.

File: `scala_dei_modelli.csv`.

## 6. Quanto costa

Mediana del tempo di addestramento di un modello, misurata sui run pubblicati.
Il rapporto non è costante: cresce al diminuire del numero di query per modello,
perché con gruppi piccoli le celle sono tante e la GAM va costruita in ognuna.

| dataset | query per modello | RTR con PDT | RTRwRuleCard con GAM | rapporto |
|---|---|---|---|---|
| FINDHR | 1 | 31.0 s | 746.9 s | 24.1× |
| FINDHR | 10 | 10.8 s | 140.7 s | 13.0× |
| FINDHRℓ | 1 | 34.0 s | 447.2 s | 13.2× |
| FINDHRℓ | 10 | 12.4 s | 88.4 s | 7.1× |

Su tutte e dieci le combinazioni il rapporto va da **7.1 a 29.2 volte**, con media
16.2. È il prezzo della GAM: guadagna poco e costa da sette a trenta volte tanto.

## 7. La selezione degli iperparametri, e una correzione del 21 settembre

La procedura: per ogni gruppo di query si provano tutte le 144 configurazioni
della griglia con una convalida a fold sui documenti di ogni query, e si tiene
quella con l'NDCG@10 medio più alto. Identica per i due modelli.

Con **tre fold** (il default del codice del gruppo) la procedura fa danno dove i
gruppi sono piccoli. Rifacendola con **dieci fold** il danno sparisce:

| dataset | \|φ\| | scelta con 3 fold, contro configurazione fissa | scelta con 10 fold, contro configurazione fissa |
|---|---|---|---|
| FINDHR | 1 | **−0.0177** (p 0.001) | −0.0040 (p 0.36) |
| FINDHR | 2 | **−0.0129** (p 0.025) | −0.0005 (p 0.67) |
| FINDHR | 10 | +0.0116 (p 0.007) | +0.0080 (p 0.024) |
| FINDHRℓ | 6 | +0.0075 (p 0.002) | +0.0070 (p 0.002) |
| FINDHRℓ | 10 | +0.0054 (p 0.005) | +0.0060 (p 0.00004) |

Con dieci fold la scelta per gruppo **non è mai significativamente peggiore**
della configurazione fissa in nessuna delle dieci combinazioni, ed è
significativamente migliore in tre. Con tre fold era significativamente peggiore
in due.

Resta però un dato che consigliamo di guardare: cambiando **solo** il numero di
fold, la configurazione scelta cambia in **90 gruppi su 100** (FINDHR) e in **99
su 100** (FINDHRℓ) quando ogni modello vede una query sola. Una procedura che
ribalta quasi tutte le sue scelte al cambiare di un dettaglio della convalida sta
in buona parte scegliendo rumore, e i guadagni stimati in validazione vanno presi
come ottimistici.

File: `tre_fold_contro_dieci.csv` (quali profondità vengono scelte) e
`tre_fold_contro_dieci_test.csv` (l'effetto sul test).

## 8. Il confronto a configurazione scelta, rifatto a dieci fold

La selezione a dieci fold è stata fatta anche sul braccio con la GAM (finita il
10 ottobre, 63 ore) e i due bracci sono stati riaddestrati con le configurazioni
scelte, cinque semi ciascuno. Questa tabella sostituisce quella provvisoria a tre
fold.

| dataset | \|φ\| | PDT | GAM | guadagno | p |
|---|---|---|---|---|---|
| FINDHR | 1 | 0.9481 | 0.9467 | −0.0014 | 0.559 |
| FINDHR | 2 | 0.9540 | 0.9551 | +0.0011 | 0.390 |
| FINDHR | 4 | 0.9572 | 0.9591 | +0.0019 | 0.488 |
| FINDHR | 6 | 0.9601 | 0.9622 | +0.0021 | 0.215 |
| FINDHR | 10 | 0.9595 | 0.9595 | −0.0001 | 0.315 |
| FINDHRℓ | 1 | 0.9780 | 0.9791 | +0.0011 | 0.758 |
| FINDHRℓ | 2 | 0.9795 | 0.9798 | +0.0002 | 0.641 |
| FINDHRℓ | 4 | 0.9766 | 0.9787 | +0.0021 | 0.386 |
| FINDHRℓ | 6 | 0.9798 | 0.9803 | +0.0006 | 0.722 |
| FINDHRℓ | 10 | 0.9801 | 0.9797 | −0.0004 | 0.602 |

**Media +0.0007, positiva in 7 combinazioni su 10, significativa in nessuna.** Il
p più piccolo è 0.215, e il guadagno medio sta sotto la deviazione standard del
rumore del seme, che è 0.0010: con la configurazione scelta per gruppo i due
bracci sono indistinguibili.

Letto braccio per braccio, la selezione rende **+0.0020 al PDT e −0.0031 alla
GAM** su FINDHR, **+0.0040 e +0.0029** su FINDHRℓ. La configurazione di
riferimento presa dal paper era quindi più vicina al meglio disponibile al
modello additivo che al meglio disponibile all'albero, e parte del +0.0038 della
sezione 3 veniva da lì.

**Attenzione a che cosa confronta questo numero**, perché non è lo stesso
confronto della sezione 3. A configurazione fissa i due bracci condividono tutto
tranne il modello di distanza, e si verifica nei file: il riferimento euclideo
misurato dentro ciascun braccio coincide fino alla quarta cifra. Con la selezione
no, perché la selezione sceglie anche la profondità del primo stadio, il numero
di vicini e il bersaglio, separatamente sui due bracci. La configurazione scelta
è identica sui due bracci in **9 gruppi su 202** su FINDHR e in **4 su 202** su
FINDHRℓ; la profondità del primo stadio coincide in 163 e 138 gruppi, il numero
di vicini in 141 e 86. Il +0.0007 confronta quindi **due sistemi regolati**, non
due distanze dentro la stessa pipeline.

Una quantità resta confrontabile fra i bracci, perché si calcola dentro la
pipeline di ciascuno: quanto il secondo stadio aggiunge al solo primo stadio.
Mediata sulle dieci combinazioni è **+0.0125 sul braccio con il PDT e +0.0155 su
quello additivo**, differenza +0.0030. Con Wilcoxon non è significativa in nessuna combinazione
(p più piccolo 0.124); con il t appaiato e il test di permutazione lo è in una,
FINDHRℓ a |φ| = 10 (+0.0281, p ≤ 0.001), dove la selezione ha dato al braccio
additivo un primo stadio più debole (0.9402 contro 0.9686) e i due modelli
completi finiscono a 0.9797 e 0.9801. La differenza viene da poche query: mediana
zero, cinque query su cento ne fanno più di metà
(`misure/confronto_modelli/tre_test_sui_confronti.py`). Il secondo stadio
additivo fa in media ancora un po' più lavoro, in parte per compensare un primo
stadio più debole; quello che il confronto complessivo dice è che i due sistemi regolati arrivano
allo stesso punto, non che le due distanze siano equivalenti.

Un'ultima cosa che cambia leggendo i due regimi insieme: **che il PDT non
raggiunga il kNN euclideo dentro la cella vale alla profondità di riferimento**,
che è 4, dove sta sotto in dieci combinazioni su dieci di 0.0035 in media. Con le
configurazioni scelte, dove l'albero può essere più profondo, ci arriva anche
lui: sotto in quattro combinazioni su dieci e sopra di 0.0002 in media. È un
limite di capacità a quella profondità, non un limite del bersaglio.

Le due affermazioni da tenere insieme, senza sceglierne una sola: la distanza
additiva migliora l'ordinamento **a parità di configurazione**, in modo costante
e di poco; **non** lo migliora quando ciascun braccio è regolato sui dati.

Una cosa emersa dalla selezione e che vale la pena sapere: sul braccio additivo
la profondità dei learner (`pdt_depth`) non cambia il punteggio di convalida in
**nessuno dei 202 gruppi** dei due dataset, mentre sul braccio con il PDT lo
cambia in tutti. Le feature sono categoriche, otto su nove prendono al più tre
valori distinti, quindi un albero univariato di profondità 2 esaurisce gli split
disponibili su quasi tutte le colonne della rappresentazione di coppia. Sul
braccio additivo la griglia effettiva è di 48 configurazioni e non 144.

File: `configurazione_scelta_fold10.csv`, da
`misure/confronto_modelli/configurazione_scelta_fold10.py`; l'effetto per braccio
da `misure/iperparametri/effetto_della_selezione.py`.

## 9. Cosa non è ancora deciso

1. I dataset sono due e vengono dalla stessa matrice di compatibilità. Gli altri
   (MQ2007 e seguenti) non sono bloccati da nessuno, ma costano tempo macchina.
2. Nessuna delle alternative interpretabili provate dentro la cella batte la
   media uniforme dei vicini, mentre una foresta, che non è interpretabile, rende
   più del doppio del guadagno ottenuto cambiando la distanza. Un aggregatore
   leggibile che prenda parte di quel margine resta la domanda aperta più utile.

## 10. I file di questa cartella

| file | cosa contiene |
|---|---|
| `confronto_multiseed_server.csv` | GAM contro PDT, cinque semi, dieci combinazioni |
| `distanza_o_aggregazione.csv` | guadagno cambiando la distanza contro guadagno cambiando l'aggregatore |
| `scala_dei_modelli.csv` | tutti i modelli, dal caso agli upper bound non interpretabili |
| `ablation.csv` | modello completo, senza informazione di query (RTR*), solo `r(x)`, solo `s(x)` |
| `configurazione_scelta.csv` | confronto a configurazione fissa e a configurazione scelta a tre fold, superato da quello a dieci |
| `configurazione_scelta_fold10.csv` | il confronto a configurazione scelta con la selezione a dieci fold su entrambi i bracci (§8) |
| `tre_fold_contro_dieci.csv` | come si sposta la profondità scelta passando da tre a dieci fold |
| `tre_fold_contro_dieci_test.csv` | l'effetto delle due selezioni sul test |
| `varianti_del_modello.csv` | le varianti del primo e del secondo stadio contro il riferimento |
| `risultati.ipynb` | il notebook che ricalcola i numeri dai file delle esecuzioni, con i commenti passo per passo |
