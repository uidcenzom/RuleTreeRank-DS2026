"""
Varianti di RTR per le fasi 3 e 4 della roadmap.

Il file è nostro: i wrapper del gruppo in wrappers.py non vengono toccati.
Con i parametri di default WrapperMixRTRVariante costruisce lo stesso modello di
WrapperMixRTR (distanza "pdt") o di WrapperMixRTRRuleCard (distanza "rulecard").

Le varianti disponibili:
- knn_pesato: la correzione è la media dei residui dei vicini pesata per
  1/distanza invece che uniforme, così contano anche i valori delle distanze e
  non solo quali vicini vengono scelti
- min_doc_foglia: numero minimo di documenti di training in ogni foglia del
  primo stadio, imposto allo stump usato a ogni nodo
- senza_foglie: il primo stadio resta un'unica foglia, quindi r(x) è la media del
  gruppo e la correzione dei vicini lavora su tutti i documenti insieme. È la
  variante "solo s(x)" della Figura 5 del paper
- WrapperRTRVariante al posto di WrapperMixRTRVariante: un modello di distanza
  per foglia e non per coppia (foglia, query), cioè senza l'informazione di
  query. È la variante RTR* della Figura 5
"""
import numpy as np
from RuleTree import RuleTreeRegressor
from RuleTree.stumps.regression import DecisionTreeStumpRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from ltr_utility import ModelParam, RankerModel
from ruletreerank import MixedRTR, RuleTreeRank, PairwiseDistanceTree, KNNRegFast, RuleCardPairwiseDistance


def media_pesata(distanze, valori):
    """Media dei valori pesata per 1/distanza, riga per riga.

    Come in scikit-learn con weights="distance": se in una riga ci sono vicini a
    distanza nulla, contano solo quelli. Le distanze apprese possono uscire
    leggermente negative su coppie mai viste, e vengono trattate come nulle.
    """
    nulle = distanze <= 0
    with np.errstate(divide="ignore"):
        pesi = 1.0 / distanze
    righe_nulle = nulle.any(axis=1)
    pesi[righe_nulle] = nulle[righe_nulle].astype(float)
    return (pesi * valori).sum(axis=1) / pesi.sum(axis=1)


class KNNRegPesato(KNNRegFast):
    """kNN veloce del secondo stadio con media pesata per 1/distanza.

    A differenza di KNNRegFast, quando la foglia ha al più k documenti non
    restituisce la media semplice: li usa tutti, ma pesati. Quindi la distanza
    conta anche nei gruppi foglia-query piccoli.
    """

    def predict_fast(self, x: np.ndarray) -> np.ndarray:
        if self._custom_metric_func is None:
            raise ValueError("Custom metric function not set.")

        X_train = self._fit_X
        y_train = np.asarray(self._y).ravel()
        n_query, n_train = x.shape[0], X_train.shape[0]
        k = min(self.n_neighbors, n_train)

        batch_size = max(1, self.max_pairs_per_batch // n_train)
        risultato = np.zeros(n_query, dtype=np.float64)
        for i in range(0, n_query, batch_size):
            blocco = x[i:i + batch_size]
            b = blocco.shape[0]
            idx_q = np.repeat(np.arange(b), n_train)
            idx_t = np.tile(np.arange(n_train), b)
            distanze = self._custom_metric_func.predict(blocco[idx_q], X_train[idx_t]).reshape(b, n_train)
            if k < n_train:
                vicini = np.argpartition(distanze, kth=k - 1, axis=1)[:, :k]
            else:
                vicini = np.tile(np.arange(n_train), (b, 1))
            risultato[i:i + b] = media_pesata(np.take_along_axis(distanze, vicini, axis=1), y_train[vicini])
        return risultato

    def predict_slow(self, X: np.ndarray, method="default"):
        if method == "euclidian":
            return (
                KNeighborsRegressor(n_neighbors=self.n_neighbors, metric="euclidean", weights="distance")
                .fit(self._fit_X, self._y)
                .predict(X)
            )
        return self.predict_fast(X)


def componenti(kwargs):
    """I pezzi del modello, uguali per la versione con e senza informazione di query.

    Gli stump di RuleTree rompono a caso i pareggi fra split ugualmente buoni, e
    RuleTreeRegressor non passa il proprio random_state allo stump che crea: con
    n_jobs_leaf > 1 i processi paralleli non vedono il seed globale e due run uguali
    davano risultati diversi. Il seed va quindi dato direttamente allo stump. Arriva
    anche alla GAM, attraverso il base_regressor.
    """
    seed = kwargs.get("random_state")
    rappresentazione = {
        "base_regressor": ModelParam(RuleTreeRegressor, {
            "max_depth": kwargs["pdt_depth"], "random_state": seed,
            "base_stumps": DecisionTreeStumpRegressor(max_depth=1, random_state=seed)}),
        "feature_concat": kwargs["feature_concat"],
        "feature_diff": kwargs["feature_diff"],
        "feature_sq_diff": kwargs["feature_sq_diff"],
        "subsample": kwargs["subsample"],
        "verbose": kwargs["verbose"],
    }
    match kwargs.get("distanza", "pdt"):
        case "pdt":
            distance_f = ModelParam(PairwiseDistanceTree, rappresentazione)
        case "rulecard":
            distance_f = ModelParam(RuleCardPairwiseDistance, {
                **rappresentazione,
                "learning_rate": kwargs["rulecard_lr"],
                "max_n_iter": kwargs["rulecard_max_n_iter"],
                "patience": kwargs["rulecard_patience"],
            })
        case altro:
            raise ValueError(f"distanza sconosciuta: {altro}")

    aggregazione, parametri_aggregazione = KNNRegFast, {"n_neighbors": kwargs["n_neighbors"], "n_jobs": 1}
    if kwargs.get("foresta_in_foglia", False):
        aggregazione = ForestaInFoglia
        parametri_aggregazione["seed"] = seed
    elif kwargs.get("knn_pesato", False):
        aggregazione = KNNRegPesato

    min_doc_foglia = kwargs.get("min_doc_foglia")
    stump_primo_stadio = {"max_depth": 1, "random_state": seed}
    if min_doc_foglia is not None:
        stump_primo_stadio["min_samples_leaf"] = min_doc_foglia

    # con profondità 0 l'albero del primo stadio non fa nessuno split: resta una
    # foglia sola, r(x) è la media del gruppo e i vicini si cercano fra tutti i documenti
    profondita = 0 if kwargs.get("senza_foglie", False) else kwargs["sdt_depth"]

    return dict(
        distance_f=distance_f,
        aggregation_f=ModelParam(aggregazione, parametri_aggregazione),
        base_regressor=RuleTreeRegressor(
            max_depth=profondita,
            max_leaf_nodes=kwargs["sdt_max_leaf_nodes"],
            min_samples_split=kwargs["min_samples_split"],
            random_state=seed,
            base_stumps=DecisionTreeStumpRegressor(**stump_primo_stadio)),
        dist_objective=kwargs["dist_objective"],
        verbose=kwargs["verbose"],
        n_jobs_leaf=kwargs["n_jobs_leaf"],
    )


class WrapperMixRTRVariante(MixedRTR):
    """MixedRTR con distanza, aggregazione e vincoli sulle foglie scelti dai parametri.

    Un modello di distanza per ogni coppia (foglia, query): è il modello completo.
    """

    def __init__(self, **kwargs):
        super().__init__(**componenti(kwargs))


class WrapperRTRVariante(RuleTreeRank):
    """RTR senza informazione di query, cioè RTR* della Figura 5 del paper.

    Un modello di distanza per ogni foglia e non per ogni coppia (foglia, query):
    i documenti di query diverse che cadono nella stessa foglia si fanno da vicini
    a vicenda. Serve a misurare quanto vale separare le query.
    """

    def __init__(self, **kwargs):
        super().__init__(**componenti(kwargs))


class ForestaInFoglia(KNNRegFast):
    """Upper bound del secondo stadio: una foresta dentro la cella al posto del kNN.

    Tiene l'interfaccia di KNNRegFast perché RTR costruisce l'aggregatore con quei
    parametri e poi gli chiama set_params, fit e predict. La distanza appresa non
    viene usata: la foresta guarda direttamente le feature dei documenti della cella
    per prevedere il residuo. Il modello smette quindi di essere interpretabile, ed è
    proprio questo il punto: dice quanto si lascia sul tavolo restando leggibili.
    """

    # La firma ripete per esteso quella di KNNRegFast di proposito: scikit-learn ricava
    # i parametri validi dalla firma di __init__, e RTR chiama set_params(metric=...)
    # sull'aggregatore appena costruito. Con **kwargs quei parametri non risulterebbero
    # più validi e set_params solleverebbe un errore.
    def __init__(self, n_neighbors=5, *, weights="uniform", algorithm="auto",
                 leaf_size=30, p=2, metric="minkowski", metric_params=None,
                 n_jobs=None, max_pairs_per_batch=2_000_000, n_alberi=100, seed=None):
        super().__init__(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params,
                         n_jobs=n_jobs, max_pairs_per_batch=max_pairs_per_batch)
        self.n_alberi = n_alberi
        self.seed = seed
        self.foresta_ = None

    def fit(self, X, y):
        # il fit del kNN serve comunque: RTR legge _fit_X e _y per le statistiche
        super().fit(X, y)
        self.foresta_ = RandomForestRegressor(
            n_estimators=self.n_alberi, random_state=self.seed, n_jobs=1
        ).fit(np.asarray(X), np.asarray(y).ravel())
        return self

    def predict_fast(self, x: np.ndarray) -> np.ndarray:
        return self.foresta_.predict(np.asarray(x))

    def predict_slow(self, X: np.ndarray, method="default"):
        # con method="euclidian" RTR chiede apposta il confronto con la distanza
        # euclidea, quindi lì si lascia rispondere il kNN come sempre
        if method == "euclidian":
            return super().predict_slow(X, method=method)
        return self.foresta_.predict(np.asarray(X))


class ModelloForte(RankerModel):
    """Upper bound senza foglie: un solo modello non interpretabile per gruppo di query.

    Non usa il primo stadio né la distanza appresa: allena una foresta o un gradient
    boosting su tutti i documenti del gruppo. Serve a fissare il tetto raggiungibile
    con gli stessi dati e lo stesso protocollo, rinunciando del tutto a spiegare.
    """

    def __init__(self, tipo="foresta", n_alberi=300, random_state=None, **kwargs):
        self.tipo = tipo
        if tipo == "foresta":
            self.modello = RandomForestRegressor(n_estimators=n_alberi, random_state=random_state, n_jobs=1)
        elif tipo == "boosting":
            self.modello = HistGradientBoostingRegressor(random_state=random_state)
        else:
            raise ValueError(f"tipo sconosciuto: {tipo}")

    def fit(self, X: np.ndarray, y: np.ndarray, q: np.ndarray = None, *args, **kwargs):
        self.modello.fit(np.asarray(X), np.asarray(y).ravel())
        return self

    def predict(self, X: np.ndarray, q: np.ndarray = None, *args, **kwargs) -> np.ndarray:
        return self.modello.predict(np.asarray(X))
