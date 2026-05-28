import re
import time
from datetime import datetime
from typing import Literal, Dict, Union, Tuple, Optional

import numpy as np
from RuleTree import RuleTreeRegressor
from joblib import Parallel, delayed
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.metrics import euclidean_distances

from ruletreerank import PairwiseDistanceTree, KNNRegFast
from ltr_utility import ModelParam, TreeModel, RankerModel
from ltr_utility.explain_utils import get_rules_list, get_pdt_rules_list

PDTMap = Dict[str, KNNRegFast]


def replace_feature_name(text, mapping):
    """Replace internal `X_i` feature references with human-readable feature names."""
    return re.sub(r"X_(\d+)", lambda x: mapping.get(int(x.group(1)), x.group(0)), text)

class RuleTreeRank(RankerModel):
    """
    Two-stage, interpretable-by-design Learning-to-Rank model for a single query.

    Stage 1 (shallow base regressor):
        A shallow, rule-based regressor (e.g., RuleTreeRegressor) fits (X, y), providing a coarse, interpretable
        score per item and a leaf assignment. Residuals = y - prediction are retained for local refinement.

    Stage 2 (Dist f. refinement within each leaf):
        For each leaf, we train a PairwiseDistanceTree (Dist f.) as an interpretable, symbolic distance function
        over pairs of instances (ζ transformation) and then fit a KNN regressor on residuals, using the
        Dist f. as the custom distance metric. At inference, we add the k-NN estimate of residuals to the coarse score
        to get the final prediction.
    """
    MAX_PAIRS_PER_BATCH = 20_000_00  # Max number of pairs to process per batch in Dist f. prediction

    def __init__(self, distance_f: Optional[ModelParam] = None, aggregation_f: Optional[ModelParam] = None,
                 base_regressor: Optional[TreeModel] = None,
                 dist_objective: Optional[Literal['dist', 'residuals', 'y']] = None,
                 use_precomputed: bool = False, precomputed_distances: Optional[Tuple[str, ndarray]] = None,
                 verbose: bool = False, n_jobs_leaf: int = 1):
        """
        Parameters
        ----------
        distance_f : ModelParam, default None
            The interpretable distance regressor to be trained independently inside each leaf.
            If is not provided, the "Euclidean" KNN metric is used.

        aggregation_f : ModelParam, default KNeighborsRegressor(n_neighbors=3) k-NN regressor used to
            aggregate neighbors' residuals as the correction term.

        base_regressor : TreeModel, default RuleTreeRegressor()
            The shallow, rule-based regressor used as the base model in Stage 1.

        dist_objective: {'dist', 'residuals', 'y'}, default 'dist'
            Controls the target for Dist f. training:
                - 'dist': Dist f. learns distances over features x.
                - 'residuals': Dist f. learns distances over residuals.
                - 'y': Dist f. learns distances over labels y.

        use_precomputed : bool, default False
            If True, precomputed distances are used for Dist f. training.

        precomputed_distances : Tuple[str, ndarray], default None
            A tuple containing: [objective type, precomputed distance matrix].

        verbose : bool, default False
            If True, timing statistics are collected during fit.

        n_jobs_leaf: int, default 1
            Number of jobs to run in parallel (dist f. in leaves).

        """

        super().__init__()
        self._distance_f = distance_f

        # Prototype KNN regressor to be cloned and fit per leaf on residuals.
        if aggregation_f is None:
            self._aggregation_f = ModelParam(model=KNNRegFast, param={"n_neighbors": 3})
        else:
            self._aggregation_f = aggregation_f

        # Stage 1: shallow rule tree
        if base_regressor is None:
            self._shallow_dt = RuleTreeRegressor(max_depth=3)
        else:
            assert isinstance(base_regressor,
                              TreeModel), "base_regressor must be an instance of TreeModel."
            self._shallow_dt = base_regressor

        assert dist_objective in (None, 'dist', 'residuals', 'y'), \
            "dist_objective must be one of {'dist', 'residuals', 'y'}"

        # Objective for dist target construction (features vs residuals vs labels).
        self._dist_objective = 'dist' if dist_objective is None else dist_objective

        self._use_precomputed = use_precomputed  # If True, precomputed distances are used for dist training.
        if precomputed_distances is not None:
            obj, pd = precomputed_distances
            assert obj == dist_objective and dist_objective in ('dist', 'y'), \
                "precomputed_distances does not match dist_objective or is invalid."
            if pd is not None:
                assert pd.ndim == 2 and pd.shape[0] == pd.shape[1], "precomputed_distances must be a square matrix."
                self._p_dist: Optional[ndarray] = pd  # Precomputed distances for Dist f. (if any).
            else:
                self._p_dist: Optional[ndarray] = None
        else:
            self._p_dist: Optional[ndarray] = None

        self._verbose = verbose  # If True, timing statistics are collected during fit.
        self._n_jobs_leaf = n_jobs_leaf  # Number of jobs to run in parallel (Dist f. in leafs).

        self._leaf_dist_map: PDTMap = {}  # Per-query storage (set in fit): leaf_id → KNN
        self._num_features_: Union[int, None] = None  # Number of features seen during fit

        self._sdt_fitted = False  # Flag to track if the shallow decision tree has been _fitted
        self._dist_fitted = False  # Flag to track if the distance function has been _fitted

    def fit(self, X: Union[DataFrame, ndarray], y: Union[Series, ndarray],
            q: Union[Series, ndarray], *args, **kwargs) -> 'RuleTreeRank':
        """
        Fit the shallow rule tree and the leaf-level residual correction models.

        Parameters
        ----------
        X : DataFrame or np.ndarray
            Feature matrix.
        y : Series or np.ndarray
            Target relevance scores.
        q : Series or np.ndarray
            Query IDs aligned with `X` and `y`.
        """
        X = X.to_numpy() if isinstance(X, DataFrame) else X
        y = y.to_numpy() if isinstance(y, Series) else y
        q = q.to_numpy() if isinstance(q, Series) else q

        shallow_dt = kwargs.get("shallow_dt", None)

        assert X.ndim == 2, "X must be a 2D array (n_samples, n_features)."
        assert X.shape[0] >= 1, "X must have at least one sample."
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples."
        assert q is not None and q.shape == y.shape, "q must be provided and have the same shape as y."
        assert len(self._leaf_dist_map) == 0, \
            "RuleTreeRank already contains leaf KNN models. Re-fitting will overwrite them."

        self._num_features_ = X.shape[1]

        # ---- Stage 1: train base regressor ----
        self._fit_shallow_dt(X=X, y=y, shallow_dt=shallow_dt)

        # ---- Stage 2: get residuals ----
        predict_apply_res = self._perform_residual(X=X, y=y, q=q)
        leafs_ids, residuals = predict_apply_res["leafs_ids"], predict_apply_res["residuals"]

        # ---- Stage 3: Train a distance function  ----
        self._fit_distance_function(X=X, y=y, q=q, leafs_ids=leafs_ids, residuals=residuals)

        return self

    def _fit_shallow_dt(self, X: ndarray, y: ndarray, shallow_dt: Optional[TreeModel] = None) -> Dict:
        """
        Fit the shallow decision tree regressor (RuleTreeRegressor) on the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix for the current query’s items.

        y : ndarray of shape (n_samples, )
            Ground-truth labels / relevance scores.

        shallow_dt: TreeModel, default None
            Prefitted shallow rule tree regressor.

        Returns
        -------
        Dict
            Dictionary containing timing statistics for shallow decision tree fitting.
        """
        # ---------- Train the Shallow decision treee (RuleTreeRegressor) ----------
        if shallow_dt is not None:
            self._shallow_dt = shallow_dt
            start_sdt_fit = end_std_fit = 0
            if self._verbose: print("Log ---", datetime.now(), "--- Using provided shallow RuleTreeRegressor...")
        else:
            if self._verbose: print("Log ---", datetime.now(), "--- Training the shallow RuleTreeRegressor...")

            start_sdt_fit = time.time()
            self._shallow_dt.fit(X, y)
            end_std_fit = time.time()

            if self._verbose: print("Log ---", datetime.now(), "--- End training the shallow RuleTreeRegressor...")
            # ---------- Train the Shallow decision treee (RuleTreeRegressor) ----------
        self._sdt_fitted = True
        return {
            "time_std": round(end_std_fit - start_sdt_fit, 4)
        }

    def _perform_residual(self, X: ndarray, y: ndarray, q: Optional[ndarray] = None) -> Dict:
        """
        Compute base-tree leaf assignments and residual targets.
        """
        assert self._sdt_fitted, "Shallow decision tree must be _fitted before calculating residuals."

        # ---------- Calculate the residuals  ----------
        if self._verbose: print("Log ---", datetime.now(), "--- Calculating leaf assignments and residuals...")

        start_rtr_apply_predict = time.time()

        # leafs_ids = ['Rrlrlllrllrrl' 'Rrlllrllllllr' 'Rllrrlllllrrll' ... 'Rrlllrlllllrlrl']
        leafs_ids = self._shallow_dt.apply(X)
        residuals = y - self._shallow_dt.predict(X)
        end_rtr_apply_predict = time.time()

        if self._verbose: print("Log ---", datetime.now(), "--- End calculating leaf assignments and residuals...")
        # ---------- Calculate the residuals  ----------

        return {
            "time_std_apply_predict": round(end_rtr_apply_predict - start_rtr_apply_predict, 4),
            "leafs_ids": leafs_ids,
            "residuals": residuals
        }

    def _fit_distance_function(self, X: ndarray, y: ndarray, q: ndarray, leafs_ids: ndarray, residuals: ndarray):
        """
        Fit the distance function (Dist f.) and KNN regressor on residuals for each leaf in parallel.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix for the current query’s items.

        y : ndarray of shape (n_samples, )
            Ground-truth labels / relevance scores.
        leafs_ids : ndarray of shape (n_samples, )
            Leaf assignments for each sample from the shallow decision tree.

        residuals : ndarray of shape (n_samples, )
            Residuals calculated as y - predictions from the shallow decision tree.

        """
        # ---- Stage 2: per-leaf Dist f. + KNN on residuals ----
        # ----- Prepare per-leaf tasks -----
        tasks = [
            {'leaf_id': leaf, 'mask': (leafs_ids == leaf)}
            for leaf in np.unique(leafs_ids)
        ]
        # ----- Prepare per-leaf tasks -----

        # ----- Fit Dist f. + KNN in leafs in parallel -----
        if self._verbose: print("Log ---", datetime.now(), "--- Starting Dist f. + KNN fitting in leafs...")

        start_pdf_overall = time.time()
        if self._n_jobs_leaf == 1:  # no parallel (avoid overhead)
            results = [self._fit_leaf(X, y, residuals, **t) for t in tasks]
        else:
            results = Parallel(n_jobs=self._n_jobs_leaf)(delayed(self._fit_leaf)(X, y, residuals, **t) for t in tasks)
        end_pdf_overall = time.time()

        if self._verbose: print("Log ---", datetime.now(), "--- Completed Dist f. + KNN fitting in leafs.")

        self._leaf_dist_map = {t['leaf_id']: res["aggregation_f"] for t, res in zip(tasks, results)}
        # ----- Fit Dist f. + KNN in leafs in parallel -----

        times_dist = np.array([res["time_dist"] for res in results]) if self._verbose else np.asarray([0])
        self._dist_fitted = True

        return {
            "time_dist_overall": round(end_pdf_overall - start_pdf_overall, 4),
            "time_dist_leaf": {
                "mean": round(times_dist.mean(), 4),
                "std": round(times_dist.std(), 4),
                "n": len(times_dist)
            },
            "times_dist": np.asarray([r["time_dist"] for r in results])
        }

    def _fit_leaf(self, X: ndarray, y: ndarray, residual: ndarray, leaf_id: str, mask: ndarray) -> Dict:
        """
        Fit Dist f. + KNN on residuals for a single leaf (and query).

        Parameters
        ----------
        leaf_id : str
            Identifier of the leaf being processed.

        X : ndarray of shape (n_samples, n_features)
            Feature matrix for the current query’s items.

        y : ndarray of shape (n_samples, )
            Ground-truth labels / relevance scores.

        residual : ndarray of shape (n_samples, )
            Residuals from the base regressor (y - predictions).

        mask : ndarray of shape (n_samples,)
            Indices for x, y, residual and precomputed distances.
        """

        assert X.ndim == 2 and X.shape[0] >= 1, \
            f"X must have at least one sample but got {X.shape, X.ndim, X.shape[0] >= 1}."
        assert X.shape[0] == y.shape[0] == residual.shape[0] == mask.shape[0], \
            "x, y, residual and idx_distances must have the same number of samples."
        assert y.ndim == residual.ndim == mask.ndim == 1, \
            "y, residual and idx_distances must have the same number of features."

        # --- Train KNN on residuals in this leaf using Dist f. as the distance metric ---
        knn = self._aggregation_f.model(**self._aggregation_f.param, max_pairs_per_batch=self.MAX_PAIRS_PER_BATCH)

        # READ THIS  self.dist IS A ModelParam OBJECT THAT CONTAINS THE Dist f. CLASS AND PARAMS
        dist_model, time_exec = None, 0
        if self._distance_f is not None:
            # ---------- Train Dist f. in this leaf ----------
            if self._verbose: print("Log ---", datetime.now(),
                                    f"--- Dist f. + KNN fitting in leafs leaf_id={leaf_id}, n_samples={X.shape[0]}")
            time_start_dist = time.time()

            dist_model = self._fit_pdt(X=X, y=y, residual=residual, mask=mask)

            time_end_dist = time.time()
            if self._verbose: print("Log ---", datetime.now(),
                                    f"--- End Dist f. + KNN fitting in leafs leaf_id={leaf_id}")
            # ---------- Train Dist f. in this leaf ----------

            time_exec = time_end_dist - time_start_dist

            def dist_metric(a, b):
                """Adapter used by scikit-learn KNN for one pairwise distance."""
                return dist_model.predict(a.reshape(1, -1), b.reshape(1, -1)).flatten()[0]

            knn.set_params(
                metric=dist_metric,
                algorithm="brute",
                weights="uniform"
            )
            knn.set_custom_metric(dist_model)  # Custom implementation function
        else:
            if self._verbose:
                print("Log ---", datetime.now(),
                      f"--- KNN fitting in leafs leaf_id={leaf_id}, n_samples={X.shape[0]} using euclidean metric")
            knn.set_params(
                metric="euclidean",
                algorithm="auto",
                weights="uniform"
            )

        # ---------- Train KNN in this leaf ----------
        if self._verbose: print("Log ---", datetime.now(), f"--- Starting KNN fitting in leafs leaf_id={leaf_id}...")
        time_start_knn = time.time()

        sub_X = X[mask]
        sub_residual = residual[mask]

        knn.n_neighbors = max(1, min(knn.n_neighbors, sub_X.shape[0]))

        knn.fit(sub_X, sub_residual)  # In this case we MUST copy "x" and "residual"

        time_end_knn = time.time()
        if self._verbose: print("Log ---", datetime.now(), f"--- End KNN fitting in leafs leaf_id={leaf_id}.")
        # ---------- Train KNN in this leaf ----------

        return {
            "leaf_id": leaf_id,
            "distance_f": dist_model if self._distance_f is not None else None,
            "aggregation_f": knn,
            "time_dist": round(time_exec, 4),
            "time_agg": round(time_end_knn - time_start_knn, 4),
        }

    def _fit_pdt(self, X: ndarray, y: ndarray, residual: ndarray, mask: ndarray) -> PairwiseDistanceTree:
        """
        Fit a PairwiseDistanceTree (Dist f.) on the provided data according to the specified objective.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix for the current query’s items.

        y : ndarray of shape (n_samples, )
            Ground-truth labels / relevance scores.

        residual : ndarray of shape (n_samples, )
            Residuals from the base regressor (y - predictions).

        mask : ndarray of shape (n_samples, )
            mask for x, y, residual and precomputed distances

        Returns
        -------
        PairwiseDistanceTree
            Fitted Dist f. model.
        """
        prec_dist = self._p_dist
        assert (prec_dist is None) or (prec_dist.shape[0] == prec_dist.shape[1] == X.shape[0]), \
            "Precomputed distances must match the number of samples in x."

        if self._use_precomputed:
            if prec_dist is not None:
                full_distances = prec_dist
            elif self._dist_objective == 'residuals':
                if self._verbose: print("Log ---", datetime.now(),
                                        "--- Computing euclidean distances over residuals for Dist f. training...")
                full_distances = euclidean_distances(residual.reshape(-1, 1))
            else:
                raise ValueError("Precomputed distances not provided.")
        else:
            full_distances: Union[ndarray, None] = None

        dist = self._distance_f.model(**self._distance_f.param)  # instantiate a new Dist f.

        if isinstance(dist, PairwiseDistanceTree):
            match self._dist_objective:
                case 'dist':
                    # Distances over features (pairwise_distances on x)
                    return dist.fit(X, None, distances=full_distances, mask=mask)

                case 'residuals':
                    # Distances over residuals
                    return dist.fit(X, residual, distances=full_distances, mask=mask)

                case 'y':
                    # Distances over labels y (pairwise_distances on y_sub)
                    return dist.fit(X, y, distances=full_distances, mask=mask)

                case _:
                    raise ValueError("Invalid dist_objective.")
        else:
            raise NotImplementedError("")

    def _compute_loop(self, X: ndarray, leafs: ndarray, fast: bool,
                      method: Literal["default", "euclidian"], q: Optional[ndarray] = None) -> ndarray:
        """
        Compute KNN residual corrections for all samples, leaf by leaf.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix to correct.
        leafs : np.ndarray
            Leaf IDs assigned by the shallow tree.
        fast : bool
            Whether to use the optimized custom-metric prediction path when available.
        method : {"default", "euclidian"}
            Distance mode used by the KNN correction model.
        """
        # ----- Get the kNN residual correction-----
        corrections = np.zeros(X.shape[0])
        for leaf, agg_f in self._leaf_dist_map.items():
            mask = (leafs == leaf)
            if not np.any(mask): continue
            X_query = X[mask]

            if (self._distance_f is None) or (not fast) or (method == "euclidian"):
                leaf_pred = agg_f.predict_slow(X_query, method=method)
                if leaf_pred.ndim > 1: leaf_pred = leaf_pred.ravel()
            else:
                leaf_pred = agg_f.predict_fast(X_query)

            corrections[mask] += leaf_pred
        # ----- Get the kNN residual correction-----
        return corrections

    def _predict_corrections(self, X: Union[DataFrame, ndarray], leafs: ndarray,
                             fast: bool, verbose: bool, q: Optional[ndarray] = None,
                             method: str = "default") -> Dict:
        """
        Predict and time the leaf-level KNN correction term.

        Returns
        -------
        dict
            Correction vector and elapsed KNN prediction time.
        """
        assert method in ["default", "euclidian"], "method must be 'default' or 'euclidian'."

        X = X.to_numpy() if isinstance(X, DataFrame) else X

        assert q is not None and q.shape[0] == X.shape[0], \
            "q must be provided and have the same number of samples as X."

        # ----- Get the kNN residual correction-----
        if verbose: print("Log ---", datetime.now(),
                          f"--- Predicting knn correction with RuleTreeRegressor on {X.shape[0]} samples...")
        start_knn = time.time()

        # ----- Get the kNN residual correction-----
        corrections = self._compute_loop(X, leafs, fast, method, q)
        # ----- Get the kNN residual correction-----

        end_knn = time.time()

        if verbose: print("Log ---", datetime.now(), "--- End predicting knn correction with RuleTreeRegressor...")

        return {
            "corrections": corrections,
            "time_knn": round(end_knn - start_knn, 4)
        }

    def _predict_coarse(self, X: Union[DataFrame, ndarray], verbose: bool = False) -> Dict:
        """
        Predict and time the shallow rule-tree score and leaf assignment.

        Returns
        -------
        dict
            Base predictions, leaf IDs, and elapsed shallow-tree prediction time.
        """
        # ----- Get the coarse predictions-----
        if verbose: print(f"Predicting coarse scores with RuleTreeRegressor on {X.shape[0]} samples...")
        start_shallow = time.time()

        predictions = self._shallow_dt.predict(X)
        if predictions.ndim > 1: predictions = predictions.ravel()
        leafs = self._shallow_dt.apply(X)

        end_shallow = time.time()
        if verbose: print("Log ---", datetime.now(),
                          f"--- End predicting coarse scores with RuleTreeRegressor on {X.shape[0]} samples...")

        return {
            "predictions": predictions,
            "leafs": leafs,
            "time_shallow": round(end_shallow - start_shallow, 4)
        }

    def validate_predict_input(self, X, q, output):
        """
        Validate prediction inputs and requested output mode.

        Parameters
        ----------
        X : DataFrame or np.ndarray
            Feature matrix to score.
        q : np.ndarray
            Query IDs aligned with `X`.
        output : str
            One of `full`, `score`, `corr`, or `euclidian`.
        """
        assert output in ["full", "score", "corr", "euclidian"]
        assert self._num_features_ is not None, "RuleTreeRank must be _fitted before prediction."
        assert q.shape[0] == X.shape[0], f"q({q.shape}) must have the same number of samples as X({X.shape})."
        assert q is not None, "q must be provided. "
        assert self._num_features_ == X.shape[1], \
            f"Input data must have the same number of features as during training ({self._num_features_})."
        assert self._sdt_fitted and self._sdt_fitted, \
            "Both the shallow decision tree and the distance function must be _fitted before prediction."

    def predict(self, X: Union[DataFrame, ndarray], q: Union[Series, ndarray] = None, *args, **kwargs):
        """
        Predict RuleTreeRank scores.

        Keyword options
        ---------------
        output : {"full", "score", "corr", "euclidian"}
            `full` returns base score plus correction; `score` returns only the shallow-tree
            score; `corr` returns only the KNN correction; `euclidian` uses Euclidean KNN correction.
        fast : bool
            Use the optimized custom-metric KNN path when available.
        verbose : bool
            Print timing logs.
        """
        verbose = kwargs.get("verbose", False)
        fast = kwargs.get("fast", True)
        output = kwargs.get("output", "full")  # "full", "score", "corr", "euclidian"

        self.validate_predict_input(X, q, output)

        X = X.to_numpy() if isinstance(X, DataFrame) else X

        coarse_res = self._predict_coarse(X=X, verbose=verbose)
        predictions, leafs = coarse_res["predictions"], coarse_res["leafs"]

        corrections_res = self._predict_corrections(X=X, leafs=leafs, fast=fast, verbose=verbose, q=q,
                                                    method=("euclidian" if output == "euclidian" else "default"))
        corrections = corrections_res["corrections"]

        match output:
            case "score":
                return predictions
            case "corr":
                return corrections
            case "full" | "euclidian":
                return predictions + corrections
            case _:
                raise ValueError("output must be 'score' or 'score_correction'")

    def explain_local(self, X: ndarray, columns_name: Optional[ndarray] = None) -> Dict:
        """
        Build a local explanation for one sample.

        The output includes the shallow score and path, the KNN neighbors used for
        the correction, learned PDT distances, and the matched rules. When
        `columns_name` is provided, rule text and neighbor columns are made readable.
        """

        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim == 2:
            assert X.shape[0] == 1
        else:
            raise ValueError()

        leaf_id = self._shallow_dt.apply(X)
        knn_pdt = self._leaf_dist_map[str(*leaf_id)]
        near_X, near_y = knn_pdt.get_kneighbors(X)
        rules_pdt = get_pdt_rules_list(X, near_X, knn_pdt.custom_metric_func)
        pdt_dist = knn_pdt.custom_metric_func.predict(np.tile(X, (near_X.shape[0], 1)), near_X)

        output = {
            "shallow_score": self.predict(X, q=np.zeros(1), output="score"),
            "rule_shallow": get_rules_list(x=X, model=self._shallow_dt),
            "correction": self.predict(X, q=np.zeros(1), output="corr"),
            "shallow_path": list(*leaf_id),
            "kneighbors": near_X,
            "kneighbors_score": near_y,
            "pdt_dist": pdt_dist,
            "rules_pdt": rules_pdt
        }
        if columns_name is not None:
            shallow_align = {k: v for k, v in enumerate(columns_name)}
            pdt_align = knn_pdt.custom_metric_func.construct_feat_dict(columns_name)

            output["rule_shallow"] = [
                *map(lambda x: replace_feature_name(x, shallow_align), output["rule_shallow"])
            ]
            output["rules_pdt"] = [
                [*map(lambda x: replace_feature_name(x, pdt_align), l)] for l in output["rules_pdt"]
            ]
            output["kneighbors"] = DataFrame(output["kneighbors"], columns=columns_name)

        return output

    # ------------------------------
    # Properties
    # ------------------------------
    @property
    def num_features_(self):
        """
        Get the number of features seen during fit.

        Returns
        -------
        num_features_ : int
            Number of features seen during fit.
        """
        return self._num_features_
