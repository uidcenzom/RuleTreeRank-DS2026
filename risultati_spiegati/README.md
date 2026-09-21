# Risultati spiegati

Questa cartella è una guida ai risultati per chi non lavora dentro il repository.
Contiene solo due cose: le tabelle che servono e il commento che le rende
leggibili. Ogni tabella è un CSV in questa stessa cartella, e per ognuna è
indicato da quale script di `misure/` proviene, se qualcuno volesse risalire al
calcolo.

Aggiornata al 21 settembre 2026.

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
| **aggregatore** | decide *come* si combinano i vicini per correggere il punteggio | kNN, cioè la media delle etichette dei k vicini scelti (l'originale) oppure una **foresta dentro la cella** |

Quando scriviamo «abbiamo cambiato la distanza» intendiamo il primo pezzo.
Quando scriviamo «abbiamo cambiato l'aggregazione» intendiamo il secondo. Non
sono alternative sullo stesso pezzo: sono due caselle diverse della stessa
architettura.

## 2. Glossario minimo

- **seed**: il seme del generatore casuale. Gli alberi rompono a caso i pareggi
  fra divisioni ugualmente buone, quindi lo stesso codice sugli stessi dati dà
  risultati un po' diversi a ogni esecuzione. Ogni configurazione è perciò
  eseguita **cinque volte con cinque semi diversi** e i numeri riportati sono le
  medie. Il solo cambio di seme sposta l'NDCG di circa 0.0035: è il metro con cui
  giudicare se una differenza è reale.
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

File: `confronto_multiseed_server.csv`.

## 4. Il collo di bottiglia non è la distanza, è l'aggregazione

Stesso riferimento (RTR con PDT e kNN), due modifiche alternative: cambiare la
distanza, oppure cambiare l'aggregatore.

| dataset | \|φ\| | guadagno cambiando la **distanza** | guadagno cambiando l'**aggregazione** |
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
> guadagnare, il pezzo su cui conviene lavorare è l'aggregazione, non la
> distanza».

File: `distanza_o_aggregazione.csv`.

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

Mediana del tempo di addestramento di un modello, misurata sui run pubblicati:

| dataset | RTR con PDT | RTRwRuleCard con GAM | rapporto |
|---|---|---|---|
| FINDHR | 11.7 s | 264.3 s | **22.6×** |
| FINDHRℓ | 14.7 s | 166.8 s | **11.4×** |

È il prezzo della GAM: guadagna poco e costa da dieci a venti volte tanto.

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

## 8. Cosa non è ancora deciso

1. La tabella a **configurazione scelta per gruppo** (`configurazione_scelta.csv`)
   è stata prodotta con la selezione a **tre fold**, cioè con la procedura che ora
   sappiamo difettosa. Per rifarla in modo pulito serve la selezione a dieci fold
   anche sul braccio con la GAM, che costa circa dieci volte quella con il PDT.
   Finché non è fatta, quella riga va considerata provvisoria.
2. Con la configurazione fissa la GAM è sopra il PDT di +0.0038 (10 su 10); con la
   configurazione scelta a tre fold il margine scende a +0.0020 (7 su 10). Quale
   dei due sia il confronto giusto dipende dal punto 1.

## 9. I file di questa cartella

| file | cosa contiene |
|---|---|
| `confronto_multiseed_server.csv` | GAM contro PDT, cinque semi, dieci combinazioni |
| `distanza_o_aggregazione.csv` | guadagno cambiando la distanza contro guadagno cambiando l'aggregatore |
| `scala_dei_modelli.csv` | tutti i modelli, dal caso agli upper bound non interpretabili |
| `ablation.csv` | modello completo, senza informazione di query (RTR*), solo `r(x)`, solo `s(x)` |
| `configurazione_scelta.csv` | confronto a configurazione fissa e a configurazione scelta (provvisoria, vedi §8) |
| `tre_fold_contro_dieci.csv` | come si sposta la profondità scelta passando da tre a dieci fold |
| `tre_fold_contro_dieci_test.csv` | l'effetto delle due selezioni sul test |
