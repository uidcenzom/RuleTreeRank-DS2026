"""
Modello di distanza pairwise basato su RuleCard per RuleTreeRank (RTRwRuleCard).

Sostituisce PairwiseDistanceTree nel secondo stadio di RTR. Al posto di un singolo
albero di regressione, la distanza tra coppie di documenti viene appresa da
PairwiseRuleCardGAM, cioè una RuleCard additiva costruita per boosting.

L'interfaccia è la stessa del PDT: fit costruisce il problema pairwise e allena
il modello, predict restituisce le distanze per coppie di righe. Il bersaglio è
la distanza euclidea al quadrato, calcolata sui valori z che RTR passa (i residui
del primo stadio, oppure le label), oppure sulle feature quando z non c'è, oppure
presa direttamente dalla matrice già pronta.

La classe eredita da PairwiseDistanceTree per un motivo preciso: dentro RuleTreeRank
c'è un controllo isinstance(dist, PairwiseDistanceTree) che decide come chiamare il
fit del modello di distanza. Ereditando dal PDT quel controllo continua a funzionare
e non serve modificare nemmeno una riga di RTR.
"""
import copy
import itertools
from typing import List, Optional, Union

import numpy as np
from numpy import ndarray
from sklearn.metrics import euclidean_distances

from RuleTree import RuleTreeRegressor

from ltr_utility import ModelParam
from ruletreerank.pdt import PairwiseDistanceTree
from PairwiseRuleCard.PairwiseRuleCardGAM import PairwiseRuleCardGAM


class _CopyableRuleTreeRegressor(RuleTreeRegressor):
    """RuleTreeRegressor che sopravvive alla copia profonda (deepcopy).

    Questa versione di RuleTree memorizza al suo interno un itertools.count usato
    per rompere i pareggi tra nodi. I generatori non sono copiabili, mentre la GAM
    duplica il suo estimatore di base a ogni passo di boosting con copy.deepcopy:
    senza questa correzione la copia fallirebbe sempre.

    La ridefinizione di __deepcopy__ copia tutto normalmente tranne il contatore,
    che viene ricreato da zero. È corretto perché quel contatore serve solo a
    rompere i pareggi dentro un singolo allenamento, e ogni copia viene poi
    riallenata da capo.
    """

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            new.__dict__[k] = itertools.count() if k == "tiebreaker" else copy.deepcopy(v, memo)
        return new


class RuleCardPairwiseDistance(PairwiseDistanceTree):
    """Modello di distanza pairwise interpretabile, basato su una RuleCard additiva (GAM).

    È il sostituto del PDT nello slot distance_f di RTR. Espone gli stessi metodi
    del PDT (fit, predict, get_rules) e internamente traduce le chiamate verso
    PairwiseRuleCardGAM.
    """

    def __init__(self,
                 base_regressor: Optional[Union[ModelParam, RuleTreeRegressor]] = None,
                 feature_concat: bool = False,
                 feature_diff: bool = True,
                 feature_sq_diff: bool = False,
                 subsample: Union[float, int] = 1.0,
                 verbose: bool = False,
                 learning_rate: float = 0.1,
                 max_n_iter: int = 50,
                 patience: int = 5,
                 metric: str = "euclidean",
                 random_state: Optional[int] = None,
                 strict: bool = False):
        """Costruisce l'adattatore.

        Parametri principali (gli stessi nomi del PDT, così i wrapper degli
        esperimenti passano lo stesso dizionario a entrambi i modelli):

        feature_concat: se True la coppia è rappresentata concatenando le due
            istanze [a, b]. Mappa su use_pairwise di RuleCard.
        feature_diff: se True la coppia è rappresentata dalla differenza assoluta
            |a - b|. Mappa su use_difference di RuleCard.
        feature_sq_diff: presente per compatibilità col PDT ma non supportato da
            RuleCard, quindi viene ignorato. La differenza assoluta porta comunque
            la stessa informazione di ordinamento, che è ciò che serve al kNN.
        learning_rate, max_n_iter, patience: iperparametri del boosting di RuleCard.
        strict: se True, quando l'allenamento di RuleCard fallisce oppure non produce
            alcuna regola viene sollevato un errore invece di ripiegare sulla
            distanza euclidea. Utile per validare che RuleCard sia davvero in uso.
            Il caso della cella con meno di tre istanze resta sempre un ripiego,
            perché lì un modello pairwise non è proprio definibile.
        """
        # Non chiamo super().__init__() di proposito: il PDT costruirebbe un
        # RuleTreeRegressor interno che qui non verrebbe mai usato.
        self.feature_concat = feature_concat
        self.feature_diff = feature_diff
        self.feature_sq_diff = feature_sq_diff
        if feature_sq_diff and verbose:
            print("RuleCardPairwiseDistance: feature_sq_diff non è supportato da RuleCard e viene ignorato.")

        # feature_concat corrisponde a use_pairwise, feature_diff a use_difference
        self._use_pairwise = bool(feature_concat)
        self._use_difference = bool(feature_diff)
        if not self._use_pairwise and not self._use_difference:
            # RuleCard ha bisogno di almeno una rappresentazione della coppia
            self._use_difference = True

        self.subsample = subsample
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.max_n_iter = max_n_iter
        self.patience = patience
        self.metric = metric
        self.strict = strict

        # tenuto come specifica e istanziato a ogni fit per evitare stato condiviso
        self._base_regressor = base_regressor
        if isinstance(base_regressor, ModelParam):
            self.random_state = base_regressor.param.get("random_state", random_state)
        elif base_regressor is not None:
            self.random_state = getattr(base_regressor, "random_state", random_state)
        else:
            self.random_state = random_state

        self.num_features_: Optional[int] = None
        self.gam_: Optional[PairwiseRuleCardGAM] = None
        self._offset_: float = 0.0
        # quando True predict usa la distanza euclidea al quadrato invece di RuleCard
        self._fallback: bool = False
        # motivo del ripiego, utile per contarli a fine esperimento
        # (None, "poche_istanze", "fit_fallito" oppure "modello_vuoto")
        self._fallback_reason: Optional[str] = None

    def _make_base_estimator(self) -> RuleTreeRegressor:
        """Crea l'estimatore di base da passare alla GAM a ogni allenamento.

        Se è un RuleTreeRegressor lo avvolge nella versione copiabile, così la
        GAM può duplicarlo a ogni passo di boosting senza andare in errore.
        """
        br = self._base_regressor
        if isinstance(br, ModelParam):
            if isinstance(br.model, type) and issubclass(br.model, RuleTreeRegressor):
                return _CopyableRuleTreeRegressor(**br.param)
            return br.model(**br.param)
        if isinstance(br, RuleTreeRegressor):
            return _CopyableRuleTreeRegressor(**br.get_params())
        if br is not None:
            return copy.deepcopy(br)
        return _CopyableRuleTreeRegressor(max_depth=3)

    def _pair_target(self, Xm: ndarray, z: Optional[ndarray],
                     distances: Optional[ndarray], mask: ndarray, n_full: int) -> ndarray:
        """Costruisce il bersaglio (la matrice delle distanze da apprendere).

        Segue la stessa logica del PDT:
        - se RTR passa una matrice di distanze già calcolata, si usa quella;
        - se passa z (i residui del primo stadio), il bersaglio è la distanza
          euclidea al quadrato tra i residui, cioè (z_i - z_j)^2;
        - altrimenti si usa la distanza euclidea al quadrato tra le feature.
        """
        if distances is not None:
            D = np.asarray(distances)
            if D.shape[0] == n_full:
                # è stata passata la matrice intera: tengo solo il blocco mascherato
                D = D[np.ix_(mask, mask)]
            return D
        if z is not None:
            zc = np.asarray(z)[mask].reshape(-1, 1)
            return euclidean_distances(zc, squared=True)
        return euclidean_distances(Xm, squared=True)

    def _raw_predict(self, x_a: ndarray, x_b: ndarray) -> ndarray:
        """Somma additiva della RuleCard, senza l'offset che dipende dal blocco.

        Ricostruisce la previsione della GAM (valore base più il contributo di
        ogni alberello moltiplicato per il learning rate) senza la traslazione
        finale che la GAM applica in base al minimo del blocco corrente.
        """
        g = self.gam_
        x_a = np.asarray(x_a, dtype=float)
        x_b = np.asarray(x_b, dtype=float)
        X_pairs = np.hstack([x_a, x_b])
        if g.use_difference:
            X_diff = np.abs(x_a - x_b)
            X = np.hstack([X_pairs, X_diff]) if g.use_pairwise else X_diff
        else:
            X = X_pairs
        pred = np.ones((X.shape[0],)) * g.base_prediction_
        for feat_idx, est in g.estimators_:
            pred += g.learning_rate * est.predict(X[:, feat_idx].reshape(X.shape[0], -1))
        return pred

    def _compute_offset(self, Xm: ndarray) -> None:
        """Calcola una costante additiva fissa che tiene le distanze non negative.

        La previsione grezza della RuleCard può essere negativa. Sommare una
        costante uguale a tutte le distanze non cambia l'ordine dei vicini scelti
        dal kNN, quindi non altera il risultato: serve solo a rispettare la
        convenzione distanza >= 0. La costante viene stimata una volta sola su un
        campione di coppie (al massimo 2000) per restare economica.
        """
        n = Xm.shape[0]
        if n < 2:
            self._offset_ = 0.0
            return
        rng = np.random.default_rng(0 if self.random_state is None else self.random_state)
        m = int(min(2000, n * n))
        ia = rng.integers(0, n, m)
        ib = rng.integers(0, n, m)
        raw = self._raw_predict(Xm[ia], Xm[ib])
        self._offset_ = float(max(0.0, -np.min(raw))) if raw.size else 0.0

    def fit(self, X: ndarray, z: Optional[ndarray] = None,
            distances: Optional[ndarray] = None,
            mask: Union[ndarray, None, List] = None) -> "RuleCardPairwiseDistance":
        """Allena la RuleCard sulle coppie della cella indicata dalla maschera.

        La firma è identica a quella del PDT. La maschera seleziona i documenti
        della cella (in MixedRTR una cella è l'intersezione di una foglia e di una
        query). Il bersaglio viene calcolato con _pair_target e passato alla GAM
        già pronto (mode 'precomputed'), quindi RuleCard non lo ricalcola: è così
        che garantiamo lo stesso identico bersaglio del PDT.
        """
        X = np.asarray(X)
        assert X.ndim == 2, "X deve essere una matrice bidimensionale."

        if mask is None:
            mask = np.ones(X.shape[0], dtype=bool)
        elif isinstance(mask, list):
            mask = np.asarray(mask, dtype=bool)

        self.num_features_ = X.shape[1]
        Xm = X[mask]
        self._fallback_reason = None

        # Caso 1: troppo pochi documenti nella cella. Con meno di tre istanze c'è
        # al massimo una coppia e nessun modello di distanza ha senso. Qui il
        # ripiego sulla euclidea è strutturale, non un errore, quindi non lo
        # blocchiamo mai. Va solo contato.
        if Xm.shape[0] < 3:
            self._fallback = True
            self._fallback_reason = "poche_istanze"
            self.gam_ = None
            return self

        y_pairs = self._pair_target(Xm, z, distances, mask, X.shape[0])

        base_est = self._make_base_estimator()
        self.gam_ = PairwiseRuleCardGAM(
            learning_rate=self.learning_rate,
            patience=self.patience,
            max_n_iter=self.max_n_iter,
            base_estimator=base_est,
            use_pairwise=self._use_pairwise,
            use_difference=self._use_difference,
            metric=self.metric,
            subsample=self.subsample,
            subsample_strategy="random",
            fast=None,  # evita la dipendenza opzionale interpret e resta deterministico
            n_jobs=1,
            random_state=42 if self.random_state is None else int(self.random_state),
            verbose=self.verbose,
        )

        # Caso 2: l'allenamento di RuleCard va in errore. Questo non è strutturale:
        # se strict è attivo lo facciamo emergere, altrimenti ripieghiamo.
        try:
            self.gam_.fit(Xm, mode="precomputed", pair_targets=y_pairs)
        except Exception as exc:
            if self.strict:
                raise
            if self.verbose:
                print(f"RuleCardPairwiseDistance: allenamento fallito ({exc!r}); uso la euclidea.")
            self._fallback = True
            self._fallback_reason = "fit_fallito"
            self.gam_ = None
            return self

        # Caso 3: nessun round di boosting ha migliorato, il modello è costante e
        # come distanza sarebbe inutile. Anche qui, se strict è attivo lo segnaliamo.
        if not getattr(self.gam_, "estimators_", None):
            if self.strict:
                raise RuntimeError("RuleCard non ha prodotto alcuna regola: modello costante.")
            self._fallback = True
            self._fallback_reason = "modello_vuoto"
            self.gam_ = None
            return self

        self._fallback = False
        self._compute_offset(Xm)
        return self

    def predict(self, x_a: ndarray, x_b: ndarray) -> ndarray:
        """Restituisce la distanza appresa per le coppie allineate (x_a[i], x_b[i]).

        Se la cella è finita in ripiego usa la distanza euclidea al quadrato,
        cioè lo stesso bersaglio che il modello avrebbe dovuto imparare.
        """
        x_a = np.asarray(x_a, dtype=float)
        x_b = np.asarray(x_b, dtype=float)
        assert x_a.shape == x_b.shape, "x_a e x_b devono avere la stessa forma."
        assert x_a.shape[1] == self.num_features_, \
            "Il numero di feature deve coincidere con quello visto in allenamento."

        if self._fallback or self.gam_ is None:
            # distanza euclidea al quadrato, lo stesso bersaglio del PDT
            return np.sum((x_a - x_b) ** 2, axis=1)

        return self._raw_predict(x_a, x_b) + self._offset_

    def get_rules(self, columns_names: Optional[List] = None) -> List[dict]:
        """Restituisce le regole additive: una per ogni round di boosting.

        Ogni voce è un piccolo albero costruito in un round. L'ordine dei round è
        anche l'ordine di importanza decrescente, perché a ogni round si sceglie la
        feature che riduce di più l'errore rimasto.
        """
        if self.gam_ is None or not getattr(self.gam_, "estimators_", None):
            return []

        nf = self.num_features_
        if columns_names is None:
            columns_names = [f"feat{i}" for i in range(nf)]

        labels: List[str] = []
        if self.gam_.use_pairwise:
            labels += [f"ist1 {c}" for c in columns_names]
            labels += [f"ist2 {c}" for c in columns_names]
        if self.gam_.use_difference:
            labels += [f"diff({c})" for c in columns_names]

        out = []
        for feat_idx, est in self.gam_.estimators_:
            sub = [labels[i] for i in feat_idx]
            try:
                rules = est.get_rules(columns_names=sub)
            except TypeError:
                rules = est.get_rules()
            out.append({"features": sub, "rules": rules})
        return out

    def construct_feat_dict(self, columns_names: ndarray) -> dict:
        """Mappa gli indici delle colonne pairwise ai nomi leggibili delle feature."""
        nf = self.num_features_
        d, c = {}, 0
        if self.gam_ is not None and self.gam_.use_pairwise:
            for i in range(nf):
                d[c + i] = "A_" + str(columns_names[i])
            c += nf
            for i in range(nf):
                d[c + i] = "B_" + str(columns_names[i])
            c += nf
        if self.gam_ is None or self.gam_.use_difference:
            for i in range(nf):
                d[c + i] = f"|A_{columns_names[i]} - B_{columns_names[i]}|"
            c += nf
        return d
