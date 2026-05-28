from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import ndcg_score

from ltr_utility import RankerModel
from ltr_utility.dataset.ltr_dataset import LtrDataset


def evaluate(pred: ndarray, labels: ndarray, groups_count: ndarray, k: int,
             iqr_trim: Optional[Tuple[int, int]] = None,
             weight:bool=False, aggregated:bool=True):
    """
        Evaluates the normalized discounted cumulative gain (NDCG) score for
        predictions grouped by specified group IDs. This function measures the
        ranking quality of the predicted scores compared to the true labels
        across multiple group. Each group is separately assessed, and the mean
        and standard deviation of the resulting scores are calculated.

        Parameters
        ----------
        pred : numpy.ndarray
            1D array of predicted scores or ranks output by the model.

        labels : numpy.ndarray
            1D array of true relevance labels corresponding to the predictions.

        groups_count : numpy.ndarray
            1D array containing the size of each query group. For example,
            [3, 2] means the first 3 items belong to query 1, and the next 2
            belong to query 2.

        k : int
            The cutoff rank at which to evaluate the NDCG score (e.g., NDCG@k).

        iqr_trim: IQRTrim tuple
                Optional tuple (a, b) specifying the percentiles for interquartile range
                trimming. If provided, scores outside the range defined by the a-th and
                b-th percentiles will be excluded from the mean and standard deviation
                calculations.
        Returns
        -------
        Tuple[numpy.floating, numpy.floating]
            A tuple containing the mean and standard deviation of the NDCG@k
            scores across all query group.
    """
    if pred is None:
        return 0.0, 0.0, 0.0

    split_indices = np.cumsum(groups_count)[:-1]

    y_true_splits = np.split(labels, split_indices)
    y_score_splits = np.split(pred, split_indices)

    results = np.asarray([
        ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k) if len(y_true) > 1 else 1
        for y_true, y_score in zip(y_true_splits, y_score_splits)
    ])

    if weight:
        w = np.log1p(groups_count)
        results = (w * results) / w.sum()

    if iqr_trim is not None:
        assert isinstance(iqr_trim, Tuple) and 0 <= iqr_trim[0] <= 100 and 0 <= iqr_trim[1] <= 100
        a, b = iqr_trim
        q25, q75 = np.percentile(results, [a, b])
        results = results[(results >= q25) & (results <= q75)]

    if aggregated:
        return np.mean(results), np.std(results), np.median(results)

    return results


def comparison(*comp: Tuple[str, RankerModel], dt: LtrDataset, k_values: Optional[List],
               iqr_trim: Optional[Tuple[int, int]] = None) -> DataFrame:
    results = [{
        name: evaluate(pred=model.predict(X=dt.x, q=dt.q), labels=dt.y, groups_count=dt.group_count,
                       k=k, iqr_trim=iqr_trim)
        for (name, model) in comp
    } for k in k_values]

    return pd.DataFrame(results, index=[str(k) for k in k_values])
