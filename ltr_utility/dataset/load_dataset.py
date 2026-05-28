from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from ltr_utility.synthetic import normalize
from .ltr_dataset import LtrDataset
from .parser import LoadDatasets


class DatasetName(Enum):
    """Supported dataset identifiers used by the shared loader."""

    MQ = 1
    WEB10 = 2
    FINDHR = 3
    YAHOO = 4
    FINDHRLIST = 5
    MQ2007LIST = 6
    MQ2008LIST = 7
    MQ2008 = 8

def load_by_query_dataset(base_path: Path,
                          dt_name: DatasetName,
                          get_first: Optional[int] = 500,
                          min_items: Optional[int] = 10,
                          max_item: Optional[int] = 400,
                          hold_out: Optional[Tuple[float, float, float]] = None,
                          verbose: bool = True,
                          random_state: int = 4):
    """
    Load a supported Learning-to-Rank dataset and apply the common query-level preprocessing.

    Parameters control query filtering, query truncation, per-query item caps, and an optional
    train/validation/test holdout split. Without `hold_out`, the function returns one `LtrDataset`;
    with `hold_out`, it returns `(train, valid, test, train_valid)`.
    """
    (base_path / "cache").mkdir(parents=True, exist_ok=True)

    dt = None
    match dt_name:
        # ------------------------------------------------------------------------------
        case DatasetName.MQ:
            load_dataset = LoadDatasets(
                train_path=base_path / "MQ2007/train.txt",
                valid_path=base_path / "MQ2007/vali.txt",
                test_path=base_path / "MQ2007/test.txt",
                train_cache=base_path / "cache/train_MQ2007.pkl",
                valid_cache=base_path / "cache/valid_MQ2007.pkl",
                test_cache=base_path / "cache/test_MQ2007.pkl"
            )
            dt = LtrDataset.concat(*load_dataset.get())
            if verbose: print("---- MQ2007 2007 loaded ----")
        # ------------------------------------------------------------------------------
        case DatasetName.MQ2008:
            load_dataset = LoadDatasets(
                train_path=base_path / "MQ2008/train.txt",
                valid_path=base_path / "MQ2008/vali.txt",
                test_path=base_path / "MQ2008/test.txt",
                train_cache=base_path / "cache/train_MQ2008.pkl",
                valid_cache=base_path / "cache/valid_MQ2008.pkl",
                test_cache=base_path / "cache/test_MQ2008.pkl"
            )
            dt = LtrDataset.concat(*load_dataset.get())
            if verbose: print("---- MQ2007 2008 loaded ----")
        # ------------------------------------------------------------------------------
        case DatasetName.MQ2007LIST:
            load_dataset = LoadDatasets(
                train_path=base_path / "MQ2007LIST/train.txt",
                valid_path=base_path / "MQ2007LIST/vali.txt",
                test_path=base_path / "MQ2007LIST/test.txt",
                train_cache=base_path / "cache/train_MQ2007LIST.pkl",
                valid_cache=base_path / "cache/valid_MQ2007LIST.pkl",
                test_cache=base_path / "cache/test_MQ2007LIST.pkl"
            )

            dt = LtrDataset.concat(*load_dataset.get())
            if verbose: print("---- MQ2007 2007 LIST loaded ----")
        # ------------------------------------------------------------------------------
        case DatasetName.MQ2008LIST:
            load_dataset = LoadDatasets(
                train_path=base_path / "MQ2008LIST/train.txt",
                valid_path=base_path / "MQ2008LIST/vali.txt",
                test_path=base_path / "MQ2008LIST/test.txt",
                train_cache=base_path / "cache/train_MQ2008LIST.pkl",
                valid_cache=base_path / "cache/valid_MQ2008LIST.pkl",
                test_cache=base_path / "cache/test_MQ2008LIST.pkl"
            )
            dt = LtrDataset.concat(*load_dataset.get())

            if verbose: print("---- MQ2007 2008 LIST loaded ----")
        # ------------------------------------------------------------------------------
        case DatasetName.WEB10:
            load_dataset = LoadDatasets(
                train_path=base_path / "MSLR-WEB10K/train.txt",
                valid_path=base_path / "MSLR-WEB10K/vali.txt",
                test_path=base_path / "MSLR-WEB10K/test.txt",
                train_cache=base_path / "cache/train_WEB10K.pkl",
                valid_cache=base_path / "cache/valid_WEB10K.pkl",
                test_cache=base_path / "cache/test_WEB10K.pkl"
            )
            dt = LtrDataset.concat(*load_dataset.get())

            if verbose: print("---- WEB10K loaded ----")
        # ------------------------------------------------------------------------------
        case DatasetName.FINDHR:
            # Merge the processed candidate/job splits into one query-sorted table.
            paths = [
                base_path / "FindHR/processed/train.csv",
                base_path / "FindHR/processed/dev.csv",
                base_path / "FindHR/processed/test.csv"
            ]
            load_dataset = (
                pd.concat((pd.read_csv(p) for p in paths))
                .drop(columns=["id_c", "rank"])
                .rename(columns={"id_j": "qid"})
                .sort_values("qid")
            )
            load_dataset["y"] = normalize(load_dataset["score"].values, num_bins=4)
            load_dataset = load_dataset.drop(columns="score")
            dt = LtrDataset(
                x=load_dataset.filter(like="fitness_"),
                y=load_dataset["y"],
                q=load_dataset["qid"],
                columns_name=list(
                    map(lambda x: x.replace("fitness_", ""), load_dataset.filter(like="fitness_").columns))
            )

            if verbose: print("---- FINDHR loaded ----")
        # ------------------------------------------------------------------------------
        case DatasetName.FINDHRLIST:
            # Keep the raw score for the later listwise target transformation.
            paths = [
                base_path / "FindHR/processed/train.csv",
                base_path / "FindHR/processed/dev.csv",
                base_path / "FindHR/processed/test.csv"
            ]
            load_dataset = (
                pd.concat((pd.read_csv(p) for p in paths))
                .drop(columns=["id_c", "rank"])
                .rename(columns={"id_j": "qid"})
                .sort_values("qid")
            )
            dt = LtrDataset(
                x=load_dataset.filter(like="fitness_"),
                y=load_dataset["score"],
                q=load_dataset["qid"],
                columns_name=list(
                    map(lambda x: x.replace("fitness_", ""), load_dataset.filter(like="fitness_").columns))
            )

            if verbose: print("---- FINDHR LIST loaded ----")
        case DatasetName.YAHOO:
            load_dataset = LoadDatasets(
                train_path=base_path / "Yahoo/train.txt",
                valid_path=base_path / "Yahoo/valid.txt",
                test_path=base_path / "Yahoo/test.txt",
                train_cache=base_path / "cache/train_yahoo.pkl",
                valid_cache=base_path / "cache/valid_yahoo.pkl",
                test_cache=base_path / "cache/test_yahoo.pkl"
            )
            dt = LtrDataset.concat(*load_dataset.get())
            if verbose: print("---- YAHOO loaded ----")
        # ------------------------------------------------------------------------------
        case _:
            raise ValueError("dt")

    # Remove queries that are too small for the ranking experiments.
    if min_items is not None:

        assert isinstance(min_items, int) and min_items > 0, "min_items must be an integer or None"
        dt.discard_minority_groups(min_items, inplace=True, verbose=verbose)
        if verbose: print(f"---- discard_minority_groups {min_items} queries ----")

    # Optionally keep only the first queries after deterministic dataset ordering.
    if get_first is not None and isinstance(get_first, int) and get_first > 0:

        get_first = min(len(dt.unique_q), get_first)
        dt.select_first_query(get_first, inplace=True)
        if verbose: print(f"---- Get first {get_first} queries ----")

    # Cap the number of documents per query to keep experiments comparable and bounded.
    if max_item is not None:

        assert isinstance(max_item, int) and max_item > 0, \
            "max_item must be an integer or None"
        dt.set_max_item_per_query(max_item, random_state=random_state, inplace=True)
        if verbose: print(f"---- max_item {max_item} -(determistic!) ----")

    match dt_name:
        case DatasetName.MQ2007LIST | DatasetName.MQ2008LIST | DatasetName.FINDHRLIST:
            # Convert continuous labels into per-query ordinal bins for listwise experiments.
            dt._y = (pd
                     .DataFrame({"qid": dt.q, "y": dt._y})
                     .groupby("qid")["y"]
                     .transform(lambda s: pd.qcut(s, q=min(30, s.nunique()), labels=False, duplicates="drop"))
            ).fillna(0).astype(int).to_numpy()
            print(f"---- Update the listwise ranking target  ----")


    if hold_out is not None:
        # Split each query independently, then keep a train+validation view for retraining.
        assert isinstance(hold_out, Tuple) and sum(hold_out) == 1, "hold_out must be an integer or a tuple"
        tr_size, vl_size, _ = hold_out
        train, valid, test = dt.holdout_per_query(train_size=tr_size, valid_size=vl_size, random_state=random_state)
        train_valid = LtrDataset.concat_by_query(train, valid)
        dt = train, valid, test, train_valid
        if verbose: print(f"---- Holdout {hold_out} - (determistic!) ----")

    return dt


def load_query_similarity(base_path: Path):
    """Load a saved `query_similarity.json` file as a dict of query groups."""

    qs = pd.read_json(base_path / f"query_similarity.json")
    query_groups = {
        c: qs[c].dropna().tolist() for c in qs.columns
    }
    return query_groups
