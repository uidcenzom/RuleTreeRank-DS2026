from pathlib import Path
from typing import Union, Tuple, List, NamedTuple, Optional, Generator, Iterable

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

LtRTuple = NamedTuple('LtRTuple', [('x', np.ndarray), ('y', np.ndarray), ('q', np.ndarray)])

class LtrDataset:
    """
        A dataset wrapper for Learning-to-Rank (LTR) tasks. This class provides an
        efficient way to store, manipulate, and split datasets grouped by query IDs.
    """

    def __init__(self,
                 x: Optional[Union[DataFrame, ndarray]] = None,
                 y: Optional[Union[Series, ndarray]] = None,
                 q: Optional[Union[Series, ndarray]] = None,
                 min_group_size: int = 0,
                 columns_name: Optional[Union[List, ndarray]] = None):
        """
            Initializes the LtrDataset instance.

            Parameters
            ----------
            x : Union[pd.DataFrame, np.ndarray]
                Feature matrix where each row represents a document.

            y : Union[pd.Series, np.ndarray]
                Relevance labels aligned with the rows in `x`.

            q : Union[pd.Series, np.ndarray]
                Query IDs aligned with `x` and `y`. This identifies which query
                each document belongs to. The input data should ideally be sorted
                by query ID.

            min_group_size : int, optional
                Minimum number of documents required for a query to be included in the dataset.

            columns_name: Optional[List[str]]
                List of column names for the features,  it will  be used to set the feature names.
                If `columns_name` is not provided, feature names will default
                to integer indices.
        """

        self._unique_q: ndarray = np.empty(0)
        self._group_count: ndarray = np.empty(0)
        self._columns_name: ndarray = np.empty(0)
        self._x: ndarray = np.empty((0, 0))
        self._y: ndarray = np.empty(0)
        self._q: ndarray = np.empty(0)

        # Cache to store the full Pandas DataFrame representation of the dataset
        self._dataset_cache: Optional[DataFrame] = None

        self._set(x=x, y=y, q=q, c_name=columns_name)

        if min_group_size > 0:
            self.discard_minority_groups(min_group_size, inplace=True)

    def _set(self,
             x: Union[DataFrame, ndarray],
             y: Union[Series, ndarray],
             q: Union[Series, ndarray],
             c_name: Union[List, ndarray]) -> 'LtrDataset':
        """
        Parameters
        ----------
        x : np.ndarray
            Feature matrix where each row represents a document.
        y : np.ndarray
            Relevance labels aligned with the rows in `x`.
        q : np.ndarray
            Query IDs aligned with `x` and `y`. This identifies which query
            each document belongs to. The input data should ideally be sorted
            by query ID.
        """
        self._x = np.empty((0, 0)) if x is None else np.asarray(x)
        self._y = np.empty(0) if y is None else np.asarray(y)
        self._q = np.empty(0) if q is None else np.asarray(q)

        assert self._x.shape[0] == self._y.shape[0] == self._q.shape[0], "X, y and q must have the same length."
        assert self._y.shape[0] == self._q.shape[0], "y and q must have the same shape"
        assert self._check_sorted(), "Query IDs must be sorted."

        if c_name is None:
            c_name = [*range(self._x.shape[1])]
        else:
            assert len(c_name) == self._x.shape[1]

        self._unique_q, self._group_count = self._compute_unique_group_count()
        self._dataset_cache = None
        self._columns_name = np.asarray(c_name)

        return self

    def select_first_query(self, k: int, inplace: bool = False) -> "LtrDataset":
        """
        Select the fist queries in the dataset
        Parameters
        ----------
        k : int
            The number of unique queries to select from the dataset, starting from the first query.
        inplace: bool
            If True, the filtering is applied in-place and the current instance is modified.
            If False, a new LtrDataset instance is returned with the filtered data, leaving the original
            instance unchanged.
        """
        assert isinstance(k, int) and k >= 0, "k must be a positive int"

        mask = np.isin(self._q, self._unique_q[:k])
        x = self._x[mask]
        y = self._y[mask]
        q = self._q[mask]
        c = self._columns_name

        return self._set(x=x, y=y, q=q, c_name=c) if inplace else LtrDataset(x=x, y=y, q=q, columns_name=c)

    def discard_minority_groups(self, min_group_size: int, inplace: bool = False, verbose: bool = True) -> "LtrDataset":
        """
        Discards queries that have fewer than `min_group_size` documents.

        Parameters
        ----------
        min_group_size : int
            The minimum number of documents required for a query to be retained in the dataset.

        inplace: bool
            If True, the filtering is applied in-place and the current instance is modified.
            If False, a new LtrDataset instance is returned with the filtered data, leaving the original
            instance unchanged.

        verbose: bool
            If true return logs
        """
        # Filter out queries that do not meet the minimum group size requirement
        valid_queries = self._unique_q[self._group_count >= min_group_size]
        num_discarded = len(self._unique_q) - len(valid_queries)

        if verbose: print("Filtered out {} queries with fewer than {} documents."
                          .format(num_discarded, min_group_size))

        mask = np.isin(self._q, valid_queries)

        x = self._x[mask]
        y = self._y[mask]
        q = self._q[mask]
        c = self._columns_name

        return self._set(x=x, y=y, q=q, c_name=c) if inplace else LtrDataset(x=x, y=y, q=q, columns_name=c)

    def random(self, k: int, random_state: Optional[int] = None) -> "LtrDataset":
        """
        Random select k unique queries from the dataset and return a new LtrDataset containing only the
        documents associated with those queries.

        Parameters
        ----------
        k : int
            The number of unique queries to randomly select from the dataset.

        random_state : int, optional
            Seed for the random number generator to ensure reproducibility. Defaults to None.

        Returns
        -------
        LtrDataset
            A new LtrDataset instance containing only the documents associated with the randomly selected queries.
        """
        if k >= len(self._unique_q):
            return self

        rng = np.random.default_rng(random_state)
        selected_q = rng.choice(self._unique_q, size=k, replace=False)

        mask = np.isin(self._q, selected_q)
        x = self._x[mask]
        y = self._y[mask]
        q = self._q[mask]
        c = self._columns_name

        return LtrDataset(x=x, y=y, q=q, columns_name=c)

    def _compute_unique_group_count(self) -> Tuple[ndarray, ndarray]:
        """
        Compute sorted query IDs and the number of documents belonging to each query.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Unique query IDs and their group sizes, in dataset order.
        """
        if len(self._x) == 0:
            group_count = np.array([])
            unique_q = np.array([])
        else:
            # This calculates the indices where the query ID changes and uses
            # them to determine the size of each query group.
            changes = np.concatenate(([0], np.where(self._q[:-1] != self._q[1:])[0] + 1, [len(self._q)]))
            group_count = np.diff(changes)
            unique_q = self._q[changes[:-1]]
        return unique_q, group_count

    def delete_columns(self, columns: Union[int, Iterable[int]], inplace=False):
        """
        Remove one or more feature columns from the dataset.

        Parameters
        ----------
        columns : Union[int, Iterable[int]]
            Column positions or names to remove.
        inplace : bool
            If True, update the current dataset; otherwise return a filtered copy.
        """
        assert isinstance(columns, (int, Iterable)), "columns must be int or Iterable"

        mask = np.ones(self._x.shape[1], dtype=bool)
        mask[columns] = False

        x = self._x[:, mask]
        y = self._y
        q = self._q
        c = self._columns_name[mask]

        return self._set(x=x, y=y, q=q, c_name=c) if inplace else LtrDataset(x=x, y=y, q=q, columns_name=c)

    def delete_items(self, index: Union[int, Iterable[int]], inplace=False):
        """
        Remove one or more document rows by row index.

        Parameters
        ----------
        index : Union[int, Iterable[int]]
            Row index or indices to remove.
        inplace : bool
            If True, update the current dataset; otherwise return a filtered copy.
        """
        assert isinstance(index, (int, Iterable)), "items must be int or Iterable"

        mask = np.ones(self._x.shape[0], dtype=bool)
        mask[index] = False

        x = self._x[mask]
        y = self._y[mask]
        q = self._q[mask]
        c = self._columns_name

        return self._set(x=x, y=y, q=q, c_name=c) if inplace else LtrDataset(x=x, y=y, q=q, columns_name=c)

    def delete_query(self, qid: Union[int, Iterable[int]], inplace: bool = False) -> "LtrDataset":
        """
        Deletes all documents associated with a specific query ID from the dataset.

        Parameters
        ----------
        qid : Union[int,List[int]]
            The query ID to be removed from the dataset.

        inplace: bool
            If True, the filtering is applied in-place and the current instance is modified.
            If False, a new LtrDataset instance is returned with the filtered data, leaving the original
            instance unchanged.
        """
        assert isinstance(qid, (int, Iterable)), "qid must be int, str or Iterable"

        mask = np.isin(self._q, qid, invert=True)

        x = self._x[mask]
        y = self._y[mask]
        q = self._q[mask]
        c = self._columns_name

        return self._set(x=x, y=y, q=q, c_name=c) if inplace else LtrDataset(x=x, y=y, q=q, columns_name=c)

    def filter_columns(self, columns: Union[int, Iterable[int]], inplace=False):
        """
        Keep only the selected feature columns.

        Parameters
        ----------
        columns : Union[int, Iterable[int]]
            Feature names or indices to retain.
        inplace : bool
            If True, update the current dataset; otherwise return a filtered copy.
        """
        assert isinstance(columns, (int, Iterable)), "columns must be int, str or Iterable"

        mask = np.isin(self._columns_name, columns, invert=True, assume_unique=True)
        columns_drop = self._columns_name[mask]

        return self.delete_columns(columns_drop, inplace=inplace)

    def _check_sorted(self):
        """Return True when query IDs are sorted in non-decreasing order."""
        return np.all(self._q[1:] >= self._q[:-1])

    def _sort_by_query(self, inplace: bool = False) -> 'LtrDataset':
        """
        Sorts the dataset in-place based on query IDs (self._q).

        This ensures that all documents belonging to the same query are contiguous, which is strictly required
        for the O(1) slicing logic used in grouping, iterating, and splitting the dataset.

        Parameters:
        -------
        inplace : bool
            If True, the sorting is applied in-place and the current instance is modified.
            If False, a new LtrDataset instance is returned with the sorted data, leaving the original
            instance unchanged.
        Returns
        -------
        LtrDataset
            Returns self to allow for method chaining.
        """
        dt = self.sort_by_query(x=self._x, y=self._y, q=self._q, columns_name=self._columns_name)
        return self._set(x=dt.x, y=dt.y, q=dt.q, c_name=self._columns_name) if inplace else dt

    def iterate_queries(self) -> Generator[Tuple[int, ndarray, ndarray], None, None]:

        """
        Generator that yields the data associated with each individual query.

        Each iteration returns:
        - an integer query identifier (`qid`)
        - a feature matrix for the query (`X_q`)
        - the corresponding label array (`y_q`)

        Yields
        ------
        Tuple[int, np.ndarray, np.ndarray]
            A tuple containing:
            - qid   : int
            - X_q   : np.ndarray  (feature matrix for the current query)
            - y_q   : np.ndarray  (label array for the current query)
        """
        assert self._check_sorted(), "The dataset must be sorted"

        start = 0
        for qid, size in zip(self._unique_q, self._group_count):
            end = start + size
            yield qid, self._x[start:end], self._y[start:end]
            start = end

    @staticmethod
    def _binary_mask(size: int, trues: int, random_state: Optional[int] = None):
        """
        Create a boolean mask with exactly `trues` randomly selected True values.

        Parameters
        ----------
        size : int
            Total length of the mask.
        trues : int
            Number of positions to mark as True.
        random_state : int, optional
            Seed used for reproducible sampling.
        """
        rng = np.random.default_rng(random_state)
        mask = np.zeros(size, dtype=bool)
        mask[rng.choice(size, size=trues, replace=False)] = True
        return mask

    def set_max_item_per_query(self, max_items: int, random_state: Optional[int],
                               inplace: bool = False) -> 'LtrDataset':
        """
        Filter a dataset by randomly maintaining a max_items of items per query.

        Parameters
        ----------
        max_items : int
            Max number of item per query
        random_state : int
            Random state for reproducibility
        inplace: bool
            If apply the transformation in place or not
        :return:
        """
        assert max_items > 0, "max_items must be greater than 0"

        mask = []
        for i in self._group_count:
            mask.extend(
                self._binary_mask(i, max_items, random_state) if i > max_items else np.ones(i, dtype=bool)
            )
        assert len(mask) == self.x.shape[0]

        x = self._x[mask]
        y = self._y[mask]
        q = self._q[mask]
        c = self._columns_name
        return self._set(x=x, y=y, q=q, c_name=c) if inplace else LtrDataset(x=x, y=y, q=q, columns_name=c)

    def hold_out(self, train_size: float, valid_size: float, random_seed: Optional[int] = None,
                 shuffle: bool = True) -> Tuple['LtrDataset', 'LtrDataset', 'LtrDataset']:
        """

        Parameters
        ----------
        train_size : float
            Fraction of the data to use for the training set (e.g., 0.7).
        valid_size : float
            Fraction of the data to use for the validation set (e.g., 0.15)
            The remaining fraction (1.0 - train_size - valid_size) is used for the test set.
        random_seed : int, optional
            Seed used to shuffle the data before splitting. Ensures reproducible splits if specified.
        shuffle: bool, optional
            Whether to shuffle the unique query IDs before splitting. Defaults to True.
        Returns
        -------
        Tuple[LtrDataset, LtrDataset, LtrDataset]
            A tuple containing three `LtrDataset` instances for the train, validation, and test
        """

        tr_q, vl_q = train_test_split(self.unique_q, train_size=train_size,
                                      shuffle=shuffle)

        vl_q, ts_q = train_test_split(vl_q, train_size=valid_size / (1.0 - train_size),
                                      random_state=random_seed, shuffle=shuffle)

        return self[tr_q], self[vl_q], self[ts_q]

    def holdout_per_query(self, train_size: float, valid_size: float,
                          random_state: Optional[int] = None) -> Tuple['LtrDataset', 'LtrDataset', 'LtrDataset']:
        """
            Splits the dataset into train, validation, and test sets on a per-query basis.
            Each query is processed independently to ensure no data leakage between splits.

            If a query contains fewer than 3 documents, the entire query is assigned to
            the train set to avoid splitting errors.

            Parameters
            ----------
            train_size : float
                Fraction of the data to use for the training set (e.g., 0.7).
            valid_size : float
                Fraction of the data to use for the validation set (e.g., 0.15).
                The remaining fraction (1.0 - train_size - valid_size) is used for the test set.
            random_state : int, optional
                Seed used to shuffle the data within each query before splitting.
                Ensures reproducible splits if specified. Defaults to None.

            Returns
            -------
            Tuple[LtrDataset, LtrDataset, LtrDataset]
                A tuple containing three `LtrDataset` instances for the train, validation,
                and test sets respectively. If the test set size is 0, an empty `LtrDataset`
                is returned for the test set.
            """
        test_size = 1.0 - train_size - valid_size
        has_test = test_size > 1e-5

        train_x, valid_x, test_x = [], [], []
        train_y, valid_y, test_y = [], [], []
        train_q, valid_q, test_q = [], [], []

        start = 0
        for qid, size in zip(self._unique_q, self._group_count):
            end = start + size
            # Slice instead of boolean mask for massive speedup
            x_q, y_q = self._x[start:end], self._y[start:end]
            start = end

            # Edge Case Guard: Skip splitting if a query has too few documents
            if size < 3:
                train_x.append(x_q)
                train_y.append(y_q)
                train_q.append(np.repeat(qid, size))
                continue

            # Split train vs (valid + test)
            tr_x, rem_x, tr_y, rem_y = train_test_split(
                x_q, y_q, train_size=train_size, random_state=random_state
            )

            train_x.append(tr_x)
            train_y.append(tr_y)
            train_q.append(np.repeat(qid, len(tr_y)))

            if has_test:
                # Calculate adjusted validation size for the remaining data
                valid_size_adj = valid_size / (valid_size + test_size)

                # Protect against edge cases where remaining data is 1 document
                if len(rem_x) > 1:
                    vl_x, ts_x, vl_y, ts_y = train_test_split(
                        rem_x, rem_y, train_size=valid_size_adj, random_state=random_state
                    )
                    valid_x.append(vl_x)
                    valid_y.append(vl_y)
                    valid_q.append(np.repeat(qid, len(vl_y)))
                    test_x.append(ts_x)
                    test_y.append(ts_y)
                    test_q.append(np.repeat(qid, len(ts_y)))
                else:
                    # If only 1 document remains, default it to validation
                    valid_x.append(rem_x)
                    valid_y.append(rem_y)
                    valid_q.append(np.repeat(qid, len(rem_y)))
            else:
                # No test set requested, all remaining goes to validation
                valid_x.append(rem_x)
                valid_y.append(rem_y)
                valid_q.append(np.repeat(qid, len(rem_y)))

        # Build datasets. Fall back to empty datasets if a split array is empty.
        def _build_dataset(x_list, y_list, q_list):
            """Build one split dataset from accumulated per-query arrays."""
            if not x_list: return LtrDataset()
            return LtrDataset(x=np.concatenate(x_list, axis=0),
                              y=np.concatenate(y_list, axis=0),
                              q=np.concatenate(q_list, axis=0),
                              columns_name=self._columns_name)

        return (_build_dataset(train_x, train_y, train_q),
                _build_dataset(valid_x, valid_y, valid_q),
                _build_dataset(test_x, test_y, test_q))

    def to_ltr(self, path: Optional[Union[Path, str]] = None) -> DataFrame:
        """
        Write a Learning-to-Rank dataset to file in LETOR/LibSVM LtR format.


        Parameters
        ----------
        path : Optional[pathlib.Path]
           File path or file name to write the dataset to.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'label', 'qid', 'f1' ... 'fN'.
        Example
        -------
        # X = np.array([[3, 0, 2, 2], [3, 3, 0, 0]])
        # y = np.array([0, 2])
        # q = np.array([1, 1])
        # File LETOR:
        # 0 qid:1 1:3 2:0 3:2 4:2
        # 2 qid:1 1:3 2:3 3:0 4:0
        """
        if isinstance(path, str):
            path = Path(path)

        X = self._x
        y = self._y
        q = self._q

        feature_cols = [i for i in range(1, X.shape[1] + 1)]
        df = pd.DataFrame(X, columns=feature_cols)
        df.insert(0, "qid", q)
        df.insert(0, "label", y)

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                for i in range(X.shape[0]):
                    parts = [f"{y[i]}", f"qid:{q[i]}"] + [f"{j + 1}:{'%.6g' % X[i, j]}" for j in range(X.shape[1])]
                    f.write(" ".join(parts) + "\n")
        return df

    def describe(self) -> str:
        """
        Provides a detailed string description of the dataset, including the number
        of samples, unique queries, and the distribution of documents per query.

        Returns
        -------
        str
            A multi-line string describing the dataset.
        """
        description = f"Total Samples: {len(self)}\n"
        description += f"Unique Queries: {len(self.unique_q)}\n"
        description += "Documents per Query:\n"
        for qid, count in zip(self.unique_q, self.group_count):
            description += f"  Query ID {qid}: {count} documents\n"
        return description

    def __add__(self, other: 'LtrDataset') -> 'LtrDataset':
        """
            Overloads the '+' operator to concatenate two LtrDataset instances.

            Parameters
            ----------
            other : LtrDataset
                Another LtrDataset instance to concatenate with the current one.

            Returns
            -------
            LtrDataset
                A new LtrDataset containing the combined data.

        """
        if not isinstance(other, LtrDataset):
            raise TypeError("Can only merge with another LtrDataset.")
        return self.concat(self, other)

    def __len__(self) -> int:
        """
            Returns the total number of documents (rows) in the dataset.
        """
        return len(self._x)

    def __getitem__(self, item: Union[Iterable[int], np.ndarray, slice, int]) -> 'LtrDataset':
        """
            Retrieves a subset of the dataset based on query IDs.

            Parameters
            ----------
            item : Union[int, List[int], np.ndarray]
                A single query ID or an iterable of query IDs to filter by.

            Returns
            -------
            LtrDataset
                A new LtrDataset instance containing only the documents that belong to the specified query IDs.
        """
        if isinstance(item, int):
            mask = self._q == item

        elif isinstance(item, slice):
            mask = np.isin(self._q, self._unique_q[item])

        elif isinstance(item, (list, ndarray)):
            mask = np.isin(self._q, np.unique(item))

        else:
            raise ValueError("Index must be an int or a list of ints representing query IDs.")

        return LtrDataset(x=self._x[mask], y=self._y[mask], q=self._q[mask], columns_name=self._columns_name)

    def __str__(self) -> str:
        """
        Returns a string representation of the dataset showing the total number
        of samples and unique queries.
        """
        return f"LtrDataset(samples={len(self)}, queries={len(self.group_count)})"

    @classmethod
    def from_tuple(cls, t: LtRTuple) -> 'LtrDataset':
        """Create an `LtrDataset` from an `(x, y, q)` named tuple."""
        return cls(t.x, t.y, t.q)

    @classmethod
    def from_LETOR(cls, dt: pd.DataFrame) -> 'LtrDataset':
        """
            Creates an LtrDataset instance from a DataFrame in LETOR format.

            The DataFrame is expected to have the following columns:
            - 'qid': Query ID for each document.
            - 'label': Relevance label for each document.
            - Feature columns: All other columns are treated as features.

            Parameters
            ----------
            dt : pd.DataFrame
                A DataFrame containing the LETOR-formatted data.

            Returns
            -------
            LtrDataset
                An instance of LtrDataset containing the parsed data.
        """
        if 'qid' not in dt.columns or 'label' not in dt.columns:
            raise ValueError("DataFrame must contain 'qid' and 'label' columns.")

        q = dt['qid'].values
        y = dt['label'].values
        x = dt.drop(columns=['qid', 'label']).values

        return cls(x=x, y=y, q=q, columns_name=list(dt.columns))

    @classmethod
    def concat(cls, *dt: 'LtrDataset') -> 'LtrDataset':
        """
            Concatenates multiple LtrDataset instances along the row axis (axis=0).

            Parameters
            ----------
            *dt : LtrDataset
                Variable length argument list of LtrDataset instances to concatenate.

            Returns
            -------
            LtrDataset
                A new LtrDataset containing the combined data from all provided instances.

        """
        if not dt:
            return LtrDataset(np.array([]), np.array([]), np.array([]))

        assert all(np.array_equal(i.feature_names(), dt[0].feature_names()) for i in dt), \
            "All dataset must have the same columns name"

        new_x = np.concatenate([d.x for d in dt], axis=0)
        new_y = np.concatenate([d.y for d in dt], axis=0)
        new_q = np.concatenate([d.q for d in dt], axis=0)
        c = dt[0].feature_names()
        return LtrDataset.sort_by_query(x=new_x, y=new_y, q=new_q, columns_name=c)

    @classmethod
    def concat_by_query(cls, *dt: 'LtrDataset') -> 'LtrDataset':
        """
        Example:
            d1.q = [1,1,2] ; d2.q = [1,3,3]
            -> unique_q = [1,2,3]
            -> q = [1,1, (d1) , 1 (d2),  2 (d1),  3,3 (d2)]

        Returns
        -------
        LtrDataset:
                A new LtrDataset containing the combined data from all provided instances.
.
        """
        if not dt or not (nonempty := [d for d in dt if len(d) > 0]):
            return LtrDataset(np.array([]), np.array([]), np.array([]))

        assert nonempty[0].x.ndim == 2, "The matrix must be 2D."
        assert all(np.array_equal(i.feature_names(), dt[0].feature_names()) for i in dt), \
            "All dataset must have the same columns name"

        n_features = nonempty[0].x.shape[1]
        for d in nonempty[1:]:
            if d.x.ndim != 2 or d.x.shape[1] != n_features:
                raise ValueError("All datasets must have the same number of features.")

        per_dataset_slices = []
        for d in dt:
            q_to_slice = {}
            if len(d) > 0:
                start = 0
                for qid, size in zip(d.unique_q, d.group_count):
                    end = start + size
                    q_to_slice[qid] = slice(start, end)
                    start = end
            per_dataset_slices.append(q_to_slice)

        union_q = np.unique(np.concatenate([d.unique_q for d in nonempty], axis=0))

        new_x, new_y, new_q = [], [], []
        for qid in union_q:
            for d, qmap in zip(dt, per_dataset_slices):
                sl = qmap.get(qid, None)
                if sl is not None:
                    new_x.append(d.x[sl])
                    new_y.append(d.y[sl])
                    new_q.append(d.q[sl])

        if not new_x:
            return LtrDataset(np.array([]), np.array([]), np.array([]))

        return LtrDataset(
            x=np.concatenate(new_x, axis=0),
            y=np.concatenate(new_y, axis=0),
            q=np.concatenate(new_q, axis=0),
            columns_name=dt[0].feature_names()
        )

    @property
    def dataset(self) -> DataFrame:
        """
            Returns a Pandas DataFrame representation of the dataset.

            Returns
            -------
            pd.DataFrame
                The dataset containing query IDs ('qid'), labels ('y'), and features.
        """
        if self._dataset_cache is None:
            dt = pd.DataFrame(self._x)
            dt.insert(0, "y", self._y)
            dt.insert(0, "qid", self._q)
            self._dataset_cache = dt
        return self._dataset_cache

    @property
    def x(self) -> ndarray:
        """Feature matrix with one row per document."""
        return self._x

    @property
    def y(self) -> ndarray:
        """Relevance labels aligned with `x`."""
        return self._y

    @property
    def q(self) -> ndarray:
        """Query IDs aligned with `x` and `y`."""
        return self._q

    @property
    def unique_q(self) -> ndarray:
        """Unique query IDs in dataset order."""
        return self._unique_q

    @property
    def group_count(self) -> ndarray:
        """Number of documents for each query in `unique_q`."""
        return self._group_count

    def feature_names(self) -> ndarray:
        """Return the feature names associated with the columns of `x`."""
        return self._columns_name

    @staticmethod
    def q2count(q: ndarray) -> ndarray:
        """Convert a sorted query-ID vector into per-query document counts."""
        return np.diff(np.concatenate(([0], np.where(q[:-1] != q[1:])[0] + 1, [len(q)])))

    @staticmethod
    def sort_by_query(x: ndarray, y: ndarray, q: ndarray,
                      columns_name: Optional[Union[List, ndarray]] = None) -> 'LtrDataset':
        """
        Sorts the dataset by query IDs while maintaining the original order of documents within each query group.

        Parameters
        ----------
        x: ndarray
            Feature matrix where each row represents a document.

        y: ndarray
            Relevance labels aligned with the rows in `x`.

        q: ndarray
            Query IDs aligned with `x` and `y`. This identifies which query each document belongs

        columns_name: Optional[List[str]]
            List of column names for the features,  it will  be used to set the feature names.
            If `columns_name` is not provided, feature names will default
            to integer indices.
        """
        if len(x) == 0:
            return LtrDataset()

        # Use stable sorting to preserve the original order of documents
        # within the same query ID (useful if they were pre-sorted by relevance)
        sort_indices = np.argsort(q, kind='stable')

        # Apply the sorting indices to all internal arrays
        x = x[sort_indices]
        y = y[sort_indices]
        q = q[sort_indices]
        c = columns_name

        return LtrDataset(x=x, y=y, q=q, columns_name=c)
