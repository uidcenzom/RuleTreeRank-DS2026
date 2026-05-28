import time
from datetime import datetime
from typing import Optional, Literal, Tuple, Dict

import numpy as np
from joblib import Parallel, delayed
from numpy import ndarray
from pandas import DataFrame

from ltr_utility import ModelParam, TreeModel
from . import RuleTreeRank, KNNRegFast

PDTQueryMap = Dict[Tuple[str, int], KNNRegFast]


class MixedRTR(RuleTreeRank):
    def __init__(self,
                 distance_f: Optional[ModelParam] = None,
                 aggregation_f: Optional[ModelParam] = None,
                 base_regressor: Optional[TreeModel] = None,
                 dist_objective: Optional[Literal['dist', 'residuals', 'y']] = None,
                 use_precomputed: bool = False,
                 precomputed_distances: Optional[Tuple[str, ndarray]] = None,
                 verbose: bool = False,
                 n_jobs_leaf: int = 1):
        # TODO: Scrivere documentazione

        super().__init__(distance_f=distance_f,
                         aggregation_f=aggregation_f,
                         base_regressor=base_regressor,
                         dist_objective=dist_objective,
                         use_precomputed=use_precomputed,
                         precomputed_distances=precomputed_distances,
                         verbose=verbose,
                         n_jobs_leaf=n_jobs_leaf)

        self._leaf_dist_map: PDTQueryMap = {}  # Map of (leaf_id, q_id) to _fitted KNNRegressor for that leaf and query

    def _fit_distance_function(self, X: ndarray, q: ndarray, y: ndarray, leafs_ids: ndarray,
                               residuals: ndarray) -> Dict:
        # TODO: Scrivere documentazione
        # ---- Stage 2: per-leaf and query Dist f. + KNN on residuals ----
        # ----- Prepare per-leaf and query tasks -----
        tasks = []
        for leaf in np.unique(leafs_ids):
            l_mask = leafs_ids == leaf  # Select samples in the current leaf
            q_sub = q[l_mask]
            for q_id in np.unique(q_sub):
                mask = l_mask & (q == q_id)  # Select samples in the current leaf and query
                tasks.append({'leaf_id': leaf, 'q': q_id, 'mask': mask})
        # ----- Prepare per-leaf and query tasks -----

        # ----- Fit Dist f. + KNN in leafs in parallel -----
        if self._verbose: print("Log ---", datetime.now(), "--- Starting Dist f. + KNN fitting in leafs and query...")

        start_pdf_overall = time.time()
        if self._n_jobs_leaf == 1:  # no parallel (avoid overhead)
            results = [self._fit_leaf(X, y, residuals, t["leaf_id"], t["mask"]) for t in tasks]
        else:
            results = Parallel(n_jobs=self._n_jobs_leaf)(delayed(self._fit_leaf)(X, y, residuals, t["leaf_id"],
                                                                                 t["mask"]) for t in tasks)
        end_pdf_overall = time.time()

        if self._verbose: print("Log ---", datetime.now(), "--- Completed Dist f. + KNN fitting in leafs.")

        self._leaf_dist_map = {(t['leaf_id'], int(t["q"])): res["aggregation_f"] for t, res in zip(tasks, results)}
        # ----- Fit Dist f. + KNN in leafs in parallel -----

        if self._verbose:
            times_dist = (
                DataFrame([(t["q"], r["time_dist"]) for t, r in zip(tasks, results)], columns=["q", "time"])
                .groupby("q")["time"]
                .agg(mean="mean", std="std")
                .to_dict(orient="index")
            )
        else:
            times_dist = {}

        self._dist_fitted = True
        return {
            "time_dist_overall": round(end_pdf_overall - start_pdf_overall, 4),
            "times_dist": times_dist
        }

    def _compute_loop(self, X: ndarray, leafs: ndarray, fast: bool,
                      method: Literal["default", "euclidian"], q: ndarray) -> ndarray:

        # ----- Get the kNN residual correction-----
        corrections = np.zeros(X.shape[0])
        for (leaf, q_id), agg_f in self._leaf_dist_map.items():
            mask = (leafs == leaf) & (q == q_id)
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
