import json
import pickle
from pathlib import Path
from typing import List, Optional, Any, Union, Tuple, Dict

import pandas as pd
from pandas import DataFrame, concat, read_json
from sklearn.datasets import load_svmlight_file

from .ltr_dataset import LtrDataset


def load_from_cache(cache_path: Path) -> DataFrame:
    """
    Loads a DataFrame from a pickle file at the specified cache path.

    Parameters
    ----------
    cache_path : Path
        Path to the pickle file containing the cached DataFrame.
    """
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {str(cache_path)}")

    try:
        with cache_path.open("rb") as f:
            df = pickle.load(f)
        if not isinstance(df, DataFrame):
            raise ValueError("Cache at %s did not contain a DataFrame. Skipping.", str(cache_path))

    except (OSError, pickle.PickleError, EOFError) as e:
        raise RuntimeError("Failed to read cache at %s: %s", str(cache_path), e)

    return df


def save_cache(obj: Any, path: Path) -> bool:
    """
    Saves an object (e.g., a DataFrame) to a pickle file at the specified path for caching purposes.

    Parameters
    ----------
    obj : Any
        The object to be cached (e.g., a DataFrame).
    path : Path
        The file path where the object should be saved as a pickle file.
    """
    if obj is None or path is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    except (OSError, pickle.PickleError) as e:
        raise RuntimeError("Failed to write cache at %s: %s", str(path), e)

    return True

def df2ltr(df: DataFrame, output_path: Path, label_col: str, qid_col: str, feature_cols: List):
    """
    Converts a pandas DataFrame to the RankLib LETOR format and saves it to a file.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the data to convert. Must include columns for labels, query IDs, and features.
    output_path : Path
        Path to the output file where the LETOR formatted data will be saved.
    label_col : str
        Name of the column in `df` that contains the relevance labels (target variable).
    qid_col : str
        Name of the column in `df` that contains the query IDs (grouping variable).
    feature_cols : List[str]
        List of column names in `df` that contain the feature values. These will be converted
    """

    # 1. Sort by QID (RankLib prefers valid chunks of QIDs together)
    # Using 'mergesort' to maintain stability if data is already partially ordered
    df_sorted = df.sort_values(by=qid_col, kind='mergesort')

    features_list = []
    for idx, col in enumerate(feature_cols, start=1):
        features_list.append(
            df_sorted[col].apply(lambda x: f"{idx}:{x:.4f}")
        )
    features_str = concat(features_list, axis=1).agg(' '.join, axis=1)
    lines = (
            df_sorted[label_col].astype(str) +
            " qid:" + df_sorted[qid_col].astype(str) +
            " " + features_str
    )

    # Scrive su file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Successfully converted {len(df)} rows to LETOR format at: {output_path}")


def ltr2df(filepath: Path, n_features: Optional[int] = None,
           keep_sparse: bool = False) -> Tuple[Union[DataFrame, Dict], int]:
    """
    Efficiently loads an LETOR format file into a Pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the dataset file.
    keep_sparse: bool
        Keep sparse matrix or not.
    n_features : Optional[int]
        Number of features in the dataset. If None, it will be inferred from the data.
    Return
    ----------
    pd.DataFrame: A DataFrame with columns ['label', 'qid', '1', '2', ... 'n_features'].
    int: The number of features in the dataset.
    """

    # load_svmlight_file parses the 'label qid:x 1:v ...' format efficiently.
    # query_id=True extracts the 'qid' token specifically.
    X_sparse, y, q_ids = load_svmlight_file(
        filepath,
        n_features=n_features,
        query_id=True,
        zero_based=False  # start from 1 or 0
    )
    true_features = X_sparse.shape[1]
    if n_features is not None and n_features != true_features:
        raise ValueError(
            f"Specified n_features={n_features} does not match the actual number of features {true_features} in the data.")

    if keep_sparse:
        result = {
            'features_matrix': X_sparse,
            'label': y,
            'qid': q_ids,
        }
    else:
        # Convert CSR matrix to dense format.
        x_dense = X_sparse.toarray()

        # Create DataFrame from features, we generate column names 1...n_features to match the file's feature indices
        feature_cols = [str(i) for i in range(1, true_features + 1)]
        df = DataFrame(x_dense, columns=feature_cols)
        # Prepend Label and QID columns
        # Using .insert is efficient for adding columns at specific locations
        df.insert(0, 'qid', q_ids)
        df.insert(0, 'label', y)

        df['label'] = df['label'].astype(int)
        df['qid'] = df['qid'].astype(int)
        result = df

    return result, true_features


class LoadDatasets:

    def __init__(self,
                 train_path: Path, valid_path: Optional[Path] = None, test_path: Optional[Path] = None,
                 force_load: bool = False, train_cache: Optional[Path] = None, valid_cache: Optional[Path] = None,
                 test_cache: Optional[Path] = None, qid_col: str = "qid", label_col: str = "label"):
        """
        Load datasets from source files or cache, with optional forced loading and caching.
        Parameters
        ----------
        train_path : Path
            Path to the training dataset file.
        valid_path : Path
            Path to the validation dataset file.
        test_path : Path
            Path to the test dataset file.
        train_cache : Path
            Path to a pickle file for training dataset cache (read or write).
        valid_cache : Path
            Path to a pickle file for validation dataset cache (read or write).
        test_cache : Path
            Path to a pickle file for test dataset cache (read or write).
        qid_col : str, default "qid"
            Column name containing the query ID.
        label_col : str, default "label"
            Column name containing the relevance label.
        """
        self._qid_col = qid_col
        self._label_col = label_col

        self.n_features = None

        if force_load:
            print("Force loading datasets without using cache.")
            self.train, tr_feature = ltr2df(train_path)
            self.valid, vl_feature = ltr2df(valid_path, tr_feature) if valid_path is not None else (None, None)
            self.test, ts_feature = ltr2df(test_path, tr_feature) if test_path is not None else (None, None)

            if vl_feature is not None:
                assert tr_feature == vl_feature, "Train/Valid datasets have different number of features."
            if ts_feature is not None:
                assert tr_feature == vl_feature == ts_feature, "Train/Valid/Test datasets have different number of features."

            print("Saving loaded datasets to cache for future runs.")
            save_cache(self.train, train_cache)
            print("Train data saved")

            if valid_path is not None and self.valid is not None:
                save_cache(self.valid, valid_cache)
                print("Valid data saved")
            if test_path is not None and self.test is not None:
                save_cache(self.test, test_cache)
                print("Test data saved")
            self.n_features = tr_feature
        else:
            tr_feature, vl_feature, ts_feature = None, None, None

            if train_cache is not None and train_cache.exists():
                print("Loading train dataset from cache.")
                self.train = load_from_cache(train_cache)
                tr_feature = self.train.shape[1] - 2  # Assuming label and qid are the first two columns
                print("Train data loaded from cache.")

            else:
                print("Loading train dataset from source file.")
                self.train, tr_feature = ltr2df(train_path)
                save_cache(self.train, train_cache)
                print("Train data loaded and saved to cache.")

            if valid_cache is not None and valid_cache.exists():
                print("Loading valid dataset from cache.")
                self.valid = load_from_cache(valid_cache)
                vl_feature = self.valid.shape[1] - 2  # Assuming label and qid are the first two columns
                print("Valid data loaded from cache.")

            elif valid_path is not None:
                print("Loading valid dataset from source file.")
                self.valid, vl_feature = ltr2df(valid_path, tr_feature) if valid_path is not None else (None, None)
                save_cache(self.valid, valid_cache)
                print("Valid data loaded and saved to cache.")

            if test_cache is not None and test_cache.exists():
                print("Loading test dataset from cache.")
                self.test = load_from_cache(test_cache)
                ts_feature = self.test.shape[1] - 2  # Assuming label and qid are the first two columns
                print("Test data loaded from cache.")

            elif test_path is not None:
                print("Loading test dataset from source file.")
                self.test, ts_feature = ltr2df(test_path, tr_feature) if test_path is not None else (None, None)
                save_cache(self.test, test_cache)
                print("Test data loaded and saved to cache.")

            if vl_feature is not None:
                assert tr_feature == vl_feature, "Train/Valid datasets have different number of features."
            if ts_feature is not None:
                assert tr_feature == ts_feature, "Train/Valid/Test datasets have different number of features."

            self.n_features = tr_feature

    def get(self, size: Union[float, int] = None,
            random_state: int = None) -> Tuple[LtrDataset, Optional[LtrDataset], Optional[LtrDataset]]:
        """
        Get the train/valid/test datasets as LtrDataset objects, with optional query-level subsampling.
        Parameters
        ----------
        size : float | int | None
            If float in (0, 1], sample that fraction of unique queries.
            If int >= 1, sample that many unique queries.
            If None, use all queries/documents.
        random_state : int | None
            Random seed to make sampling reproducible.
        """
        return self.get_train(size, random_state), self.get_valid(size, random_state), self.get_test(size, random_state)

    def get_train(self, size: Union[float, int] = None, random_state: int = None) -> LtrDataset:
        """
        Get the training dataset as an LtrDataset, with optional query-level subsampling.
        Parameters
        ----------
        size : float | int | None
            If float in (0, 1], sample that fraction of unique queries.
            If int >= 1, sample that many unique queries.
            If None, use all queries/documents.
        random_state : int | None
            Random seed to make sampling reproducible.
        """
        assert self.train is not None, "Train dataset is not loaded. Please load it before calling get_train()."
        return self._get_source(self.train, size=size, random_state=random_state)

    def get_valid(self, size: Union[float, int] = None, random_state: int = None) -> LtrDataset:
        """
        Get the valid dataset as an LtrDataset, with optional query-level subsampling.
        Parameters
        ----------
        size : float | int | None
            If float in (0, 1], sample that fraction of unique queries.
            If int >= 1, sample that many unique queries.
            If None, use all queries/documents.
        random_state : int | None
            Random seed to make sampling reproducible.
        """
        assert self.valid is not None, "Valid dataset is not loaded. Please load it before calling get_train()."
        return self._get_source(self.valid, size=size, random_state=random_state)

    def get_test(self, size: Union[float, int] = None, random_state: int = None) -> LtrDataset:
        """
        Get the test dataset as an LtrDataset, with optional query-level subsampling.
        Parameters
        ----------
        size : float | int | None
            If float in (0, 1], sample that fraction of unique queries.
            If int >= 1, sample that many unique queries.
            If None, use all queries/documents.
        random_state : int | None
            Random seed to make sampling reproducible.
        """
        assert self.test is not None, "Test dataset is not loaded. Please load it before calling get_train()."
        return self._get_source(self.test, size=size, random_state=random_state)

    def _get_source(self, source: DataFrame, size: Union[float, int] = None, random_state: int = None) -> LtrDataset:
        """
        Prepare an LtrDataset from a raw DataFrame, with optional query-level subsampling.

        Parameters
        ----------
        source : DataFrame
            Full dataframe containing at least feature columns, label column, and qid column.
        size : float | int | None
            If float in (0, 1], sample that fraction of unique queries.
            If int >= 1, sample that many unique queries.
            If None, use all queries/documents.
        random_state : int | None
            Random seed to make sampling reproducible.

        Returns
        -------
        LtrDataset
            Wrapped dataset with X/Y/Q aligned and sorted by query id.
        """

        self._validate_columns_exist(source)
        _, label_col, qid_col = self._resolve_feature_columns()

        # Optional: subsample by queries
        if size is not None:
            if isinstance(size, float):
                if not (0.0 < size <= 1.0):
                    raise ValueError("If 'size' is float, it must be in the interval (0, 1].")
            elif isinstance(size, int):
                if size < 1:
                    raise ValueError("If 'size' is int, it must be >= 1.")
            else:
                raise ValueError("size must be float, int, or None.")

            queries = source[qid_col].drop_duplicates()
            # Edge case: requesting more queries than available
            if isinstance(size, int) and size > len(queries):
                size = len(queries)

            sample_q = (
                queries.sample(frac=size, random_state=random_state)
                if isinstance(size, float)
                else queries.sample(n=size, random_state=random_state)
            )
            sampled_df = source[source[qid_col].isin(sample_q)]
        else:
            sampled_df = source

        # Keep a deterministic ordering by qid to help group-based consumers.
        sampled_df = sampled_df.sort_values(by=qid_col)

        # Slice out X/Y/Q
        feature_cols, _, _ = self._resolve_feature_columns()
        x = sampled_df[feature_cols]
        y = sampled_df[label_col]
        q = sampled_df[qid_col]

        return LtrDataset(x=x, y=y, q=q)

    def _resolve_feature_columns(self) -> Tuple[list, str, str]:
        """
        Compute the expected feature column names, label and qid column names.

        Returns
        -------
        features, label_col, qid_col
        """
        feats = [f"{i}" for i in range(1, self.n_features + 1)]
        return feats, self._label_col, self._qid_col

    def _validate_columns_exist(self, df: DataFrame) -> None:
        """Ensure required columns are present; raise ValueError otherwise."""
        features, label_col, qid_col = self._resolve_feature_columns()

        missing = [c for c in ([label_col, qid_col] + features) if c not in df.columns]
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {missing}. "
                f"Expected features {features[:3]}... (total {len(features)}), "
                f"label '{label_col}', qid '{qid_col}'."
            )


def load_mixed_best_results(final: Path, name: str) -> DataFrame:
    df = read_json(final)

    df[["qxm", "eval_k"]] = df[["qxm", "eval_k"]].astype(int)
    df[["ndcg_mean", "ndcg_std", "ndcg_median"]] = df[name].apply(lambda x: pd.Series(x))
    df["model"] = name.replace("model_", "")

    df = df[["model", "qxm", "eval_k", "ndcg_mean", "ndcg_std", "ndcg_median"]]
    df = df.sort_values(by=["qxm", "eval_k"], ascending=[True, False]).reset_index(drop=True)
    return df
