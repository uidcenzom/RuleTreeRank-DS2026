from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union, Type

import numpy as np
import pandas as pd
from joblib import delayed, Parallel, parallel_backend
from pandas import DataFrame
from tqdm import tqdm

from ltr_utility import ModelParam, RankerModel
from ltr_utility.dataset.ltr_dataset import LtrDataset
from ruletreerank import QueryRanker
from .evaluation import comparison, evaluate
from .model_selection import save_list_to_json
from .query_kfold import QueryKFold


def train_evaluate_config(m: Type[RankerModel], conf: Dict, tr: Tuple, vl: Optional[Tuple] = None,
                          eval_at: int = 10, fold: Optional[int] = None, weight_query:bool=False) -> Union[np.floating, None]:
        assert (fold is None and vl is not None) or (fold is not None and vl is None)

        # ----------------- EVALUATE MODEL WITH A CONFIG -----------------
        if fold is not None and fold > 1:
            # if we entre here, we assume that tr = tr+vl e vl is none
            # ----------------------- k fold -------------------------------
            perf_value = np.mean([
                evaluate(
                    pred=m(**conf).fit(X=sub_tr[0], y=sub_tr[1], q=sub_tr[2]).predict(X=sub_ts[0], q=sub_ts[2]),
                    labels=sub_ts[1],
                    groups_count=LtrDataset.q2count(sub_ts[2]),
                    k=eval_at,
                    weight=weight_query)[0]
                for sub_tr, sub_ts in QueryKFold(n_splits=fold).split(*tr)
            ])
            # ----------------------- k fold -------------------------------
        else:
            # if we enter here we assume tr is not none e vl is not none
            # ----------------------- hold out -------------------------------
            perf_value = evaluate(
                pred=m(**conf).fit(X=tr[0], y=tr[1], q=tr[2]).predict(X=vl[0], q=vl[2]),
                labels=vl[1],
                groups_count=LtrDataset.q2count(vl[2]),
                k=eval_at,
                weight=weight_query)[0]
            # ----------------------- hold out -------------------------------

        # ----------------- EVALUATE MODEL WITH A CONFIG -----------------
        output = perf_value

        return output


def train_model_selection(ranker: ModelParam, query_group: List[List], train: LtrDataset,
                          valid: Optional[LtrDataset] = None, eval_at: int = 10, weight_query:bool=False,
                          max_workers: int = 1, num_fold: Optional[int] = None):
    assert eval_at > 0 and max_workers > 0
    assert isinstance(ranker.model, RankerModel)
    assert isinstance(ranker.param, list) and len(ranker.param) > 0
    assert (num_fold is None and valid is not None) or (num_fold is not None and valid is None)

    ranker_cls, configs = ranker
    max_workers = min(max_workers, len(configs))

    list_confs = []
    for queries_batch in tqdm(query_group, desc="Model Selection"):
        tr_sub = train[queries_batch]
        vl_sub = valid[queries_batch] if valid is not None else None

        task_parameters = [{
            "m": ranker_cls,
            "conf": conf,
            "eval_at": eval_at,
            "tr": (tr_sub.x, tr_sub.y, tr_sub.q),
            "vl": (vl_sub.x, vl_sub.y, vl_sub.q) if vl_sub is not None else None,
            "fold": num_fold,
            "weight_query": weight_query
        } for conf in configs]

        if max_workers == 1:
            results = [train_evaluate_config(**t) for t in task_parameters]
        else:
            with parallel_backend("loky", inner_max_num_threads=1):
                results = (Parallel(n_jobs=max_workers, batch_size="auto")
                           (delayed(train_evaluate_config)(**t) for t in task_parameters))
            assert len(configs) == len(results)

        best_idx = np.argmax([res for res in results])
        list_confs.append((queries_batch, configs, best_idx, results))

    return list_confs


def query_model_selection(q_per_model: List[int], ranker: ModelParam, train: LtrDataset,
                          query_groups: Optional[Dict[int, List]] = None, num_fold: Optional[int] = 3,
                          valid: Optional[LtrDataset] = None, max_workers: Union[int,List] = 1,
                          weight_query:bool=False, save_cache: bool = False):
    if query_groups is not None:
        assert all(i in query_groups.keys() for i in q_per_model)

    if isinstance(max_workers, List):
        assert len(max_workers) == len(q_per_model)
        max_workers = {k:v for k, v in zip(q_per_model, max_workers)}
    else:
        max_workers = {k: max_workers for k in q_per_model}

    assert isinstance(ranker.model, RankerModel)
    assert isinstance(ranker.param, list) and len(ranker.param) > 0
    assert q_per_model and all(isinstance(i, int) and i > 0 for i in q_per_model), \
        "_q_per_model must be a list of positive integers."

    dt_ms = []
    for i in q_per_model:

        if query_groups is not None:
            q_groups = query_groups[i]
        else:
            q_groups = np.array_split(train.unique_q, int(np.ceil(len(train.unique_q) / i)))

        list_confs = train_model_selection(
            ranker=ranker,
            query_group=q_groups,
            train=train,
            valid=valid,
            eval_at=10,
            max_workers=max_workers[i],
            num_fold=num_fold,
            weight_query=weight_query
        )
        if save_cache: save_list_to_json(list_confs, f"cache_{i}_{str(ranker.model)}.json")

        t = pd.DataFrame(list_confs)
        t.insert(0, "qxm", i)
        dt_ms.append(t)

    dt = pd.concat(dt_ms).rename(columns={0: "query", 1: "configs", 2: "idx_best", 3: "results"})
    return dt


def train_query_based_ltr(q_per_model: List[int], ranker: Type[RankerModel], train: LtrDataset,
                          init_param: Optional[Dict[str, Any]] = None, set_group: bool = True, set_q: bool = False):

    if init_param is None: init_param = {}

    assert q_per_model and all(isinstance(i, int) and i > 0 for i in q_per_model), \
        "_q_per_model must be a list of positive integers."

    models = {}
    for i in q_per_model:
        models[i] = QueryRanker(
            ranker=ModelParam(model=ranker, param=init_param),
            q_per_model=i,
            set_group=set_group,
            set_q=set_q
        ).fit(train=train)

    return models


def test_query_based_ltr(*models: Tuple[str, Dict[int, Type[RankerModel]]], iqr_trim: Optional[Tuple[int, int]] = None,
                         q_per_model: List[int], dt: LtrDataset, k: Union[int, List[int]]) -> pd.DataFrame:
    if isinstance(k, int):
        k = [k]

    list_results: List[DataFrame] = []
    print("Evaluating models...")
    for i in tqdm(q_per_model, total=len(q_per_model), desc="Evaluating models..."):
        q_models = [(name, model[i]) for name, model in models]
        q_result = (
            comparison(*q_models, dt=dt, k_values=k, iqr_trim=iqr_trim)
            .add_prefix("model_", axis=1)
        )
        q_result.insert(0, "qxm", i)
        list_results.append(q_result)

    df_results = pd.concat(list_results, axis=0).reset_index().rename(columns={"index": "eval_k"})

    return df_results


def extract_query_configs(path: Path):
    ms_df = pd.read_json(path)[["qxm", "query", "configs", "idx_best"]]
    ms_df["configs"] = ms_df[["configs", "idx_best"]].apply(lambda x: x["configs"][x["idx_best"]], axis=1)
    grouped = (
        ms_df.drop(columns=["idx_best"]).groupby("qxm")
        .apply(lambda g: (list(g["query"]), list(g["configs"])))
        .to_dict()
    )
    return grouped


def final_retraining_evaluate(model: RankerModel, train: LtrDataset, test: LtrDataset, ms_result: Path, name: str,
                              eval_at: List[int]):
    configs = extract_query_configs(ms_result)
    q_per_model = list(configs.keys())

    models = {
        i: QueryRanker(ranker=ModelParam(model=model, param=configs[i]), q_per_model=i).fit(train=train)
        for i in q_per_model
    }
    comp_dict_test = test_query_based_ltr((name, models), q_per_model=q_per_model, dt=test, k=eval_at)

    (ms_result.parent / "final").mkdir(parents=True, exist_ok=True)
    comp_dict_test.to_json(ms_result.parent / "final" / f"{name}_{str(q_per_model)}.json", orient="records", indent=4)
