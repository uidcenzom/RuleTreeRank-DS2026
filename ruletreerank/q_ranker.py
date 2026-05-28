import collections
import warnings
from typing import Dict, List, Union, Optional, Type

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from ltr_utility import ModelParam, RankerModel
from ltr_utility.dataset.ltr_dataset import LtrDataset


class QueryRanker(RankerModel):
    """
    A meta-model that trains multiple ranking models, each responsible for a subset of queries.
    """

    def __init__(self, ranker: ModelParam, q_per_model: int = 1, batch_query: Optional[List[List]] = None,
                 *args, **kwargs):
        """
        Parameters
        ----------
        ranker : ModelParam
            Ranker class and parameter dictionary used for each query batch.
        q_per_model : int
            Maximum number of query IDs assigned to one fitted ranker.
        batch_query : list of list, optional
            Explicit query batches. If omitted, batches are built from sorted query IDs.
        """
        super().__init__(*args, **kwargs)
        assert q_per_model > 0, "q_per_model must be a positive integer."
        assert isinstance(ranker.model, RankerModel)

        if batch_query is not None:
            assert isinstance(batch_query, list) and all(isinstance(b, list) for b in batch_query), \
                "batch_query must be a list of lists."

        self._ranker = ranker
        self._q_per_model = q_per_model
        self._batch_query = batch_query

        self._query_to_model: Dict[int, RankerModel] = {}
        self._models_to_qs: Dict[RankerModel, List[int]] = {}
        self._fitted = False

    def fit(self, X: np.ndarray = None, y: np.ndarray = None, q: np.ndarray = None, *args, **kwargs) -> 'QueryRanker':
        """
        Fit one ranker per query batch.

        Notes
        -----
        The training dataset is read from `kwargs["train"]`; positional `X`, `y`, and `q`
        are kept only for compatibility with the `RankerModel` protocol.
        """
        assert "train" in kwargs, "fit must contain 'train'"

        train = kwargs["train"]
        assert isinstance(train, LtrDataset), "Train must be a LtrDataset"
        assert not self._fitted, "QueryRanker is already fitted. Create a new instance to fit again."
        assert np.all(np.diff(train.q) >= 0), "Train q_ids must be sorted in non-decreasing order."

        self._q_per_model = min(len(train.unique_q), self._q_per_model)
        self._query_to_model.clear()

        print(f"Training with one-shot mode ({type(self._ranker.param).__name__} params).")
        self._fit_one_shot(train=train)

        # Build mapping for predictions
        model_groups = collections.defaultdict(list)
        for q, model in self._query_to_model.items():
            model_groups[model].append(q)
        self._models_to_qs = dict(model_groups)

        self._fitted = True
        return self

    def _get_batches(self, unique_qs: ndarray) -> List[ndarray]:
        """
        Split sorted query IDs into batches of at most `q_per_model` queries.

        Parameters
        ----------
        unique_qs : np.ndarray
            Query IDs available in the training dataset.
        """
        counts = len(unique_qs)
        result = np.array_split(unique_qs, int(np.ceil(counts / self._q_per_model)))
        return [i.tolist() for i in result]

    def _fit_one_shot(self, train: LtrDataset) -> None:
        """
        Train all query-batch rankers and populate the query-to-model mapping.

        Parameters
        ----------
        train : LtrDataset
            Sorted training dataset used to extract query subsets.
        """
        if self._batch_query is not None:
            aligned = len(self._batch_query) == int(np.ceil(len(train.unique_q) / self._q_per_model))
            if not aligned: warnings.warn("Mismatch between query batches and configs.")
            batches = self._batch_query
        else:
            batches = self._get_batches(train.unique_q)

        model_cls: Type[RankerModel] = self._ranker.model
        params: Union[Dict, List] = self._ranker.param

        list_conf = params if isinstance(params, list) else [params] * len(batches)
        assert len(batches) == len(list_conf), "Mismatch between query batches and configs."
        assert set(sum(batches, [])).issubset(train.unique_q)

        for queries_batch, conf in tqdm(zip(batches, list_conf), total=len(batches), desc="Training"):

            sub_dt = train[queries_batch]

            q_ranker = model_cls(**conf)
            q_ranker.fit(X=sub_dt.x, y=sub_dt.y, q=sub_dt.q)

            for q in queries_batch:
                self._query_to_model[int(q)] = q_ranker

    def predict(self, X: ndarray, q: ndarray, *args, **kwargs) -> ndarray:
        """
        Predict scores by dispatching each row to the ranker trained for its query ID.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix to score.
        q : np.ndarray
            Query IDs aligned with `X`.
        """
        assert self._fitted, "Model must be fitted before prediction."
        assert q.shape[0] == X.shape[0], "q must be provided and match the length of X."

        missing_queries = np.setdiff1d(np.unique(q), list(self._query_to_model.keys()))
        assert missing_queries.size <= 0, \
            f"No models found for these query IDs: {missing_queries}"

        y_pred = np.zeros(len(X))

        for model, qs in self._models_to_qs.items():
            mask = np.isin(q, qs)
            if np.any(mask): y_pred[mask] = model.predict(X=X[mask], q=q[mask], **kwargs)

        return y_pred
