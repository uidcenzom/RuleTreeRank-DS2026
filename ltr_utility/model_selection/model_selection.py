import itertools
import json
import os
import pickle
import time
from abc import abstractmethod, ABC
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from numpy import ndarray, floating, integer
from pandas import DataFrame

from ltr_utility import RankerModel
from ltr_utility.dataset.ltr_dataset import LtrDataset
from .evaluation import evaluate


class WrapperModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = config
        self.hash = hash(config.values())

        self.model = None
        self.logs: Dict[str, Any] = {}
        self.has_fitted: bool = False

    @abstractmethod
    def fit(self, t: LtrDataset) -> 'WrapperModel':
        pass

    def predict(self, dt: LtrDataset, *args, **kwargs) -> ndarray:
        assert self.has_fitted, "Model has not been _fitted yet. Call fit() before prediction."
        return self.model.predict(dt.x)

    def get_model(self) -> RankerModel:
        return self.model

    def get_logs(self) -> Dict[str, Any]:
        return self.logs

    def save_model(self, file_path: Path) -> None:
        assert self.has_fitted, "Model has not been _fitted yet. Call fit() before saving."
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)


def make_grid_search(params: Dict[str, List[Any]],
                     fixed_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """

    Parameters
    ----------
    params : Dict[str, List[Any]]
        Hyperparameter grid to search over.
    fixed_params : Dict[str, Any]
        Fixed parameters to include in every configuration.

    Returns
    -------
    List[Dict[str, Any]]
        List of all hyperparameter configurations (as dictionaries).
    """
    if fixed_params is None:
        fixed_params = {}

    all_configs = []
    for i in itertools.product(*params.values()):
        all_config = {
            **{k: v for k, v in zip(params.keys(), i)},
            **fixed_params,
        }
        all_configs.append(all_config)

    return all_configs

def compute_configurations_to_do(base_result: Path, all_config: List[Dict]):
    """
    Compare executed configurations inside base_result with all_config,
    and return which configurations still need to be done.

    Parameters
    ----------
    base_result : Path
        Directory containing JSON files with executed configurations.
    all_config : list of dict
        Full list of all configurations (may contain 'hash').

    Returns
    -------
    dict with:
        - done_configurations : list of dict
        - config_to_do : list of dict
    """

    # Extract done configurations from JSON files
    done_configurations = []
    name_done_configurations = []
    for i in base_result.iterdir():
        if i.is_file():
            data = json.loads(i.read_text())
            # keep only keys starting with conf_, removing the prefix
            only_conf = {k.replace("conf_", ""): v for k, v in data.items() if k.startswith("conf_")}
            done_configurations.append(only_conf)
            name_done_configurations.append(i.name)

    # Remove 'hash' for comparison
    all_config_ = [{k: v for k, v in d.items() if k != "hash"} for d in all_config]

    # Identify missing configurations
    config_to_do = [
        original for original, stripped in zip(all_config, all_config_)
        if stripped not in done_configurations
    ]

    return {
        "names_done_configurations": name_done_configurations,
        "done_configurations": done_configurations,
        "config_to_do": config_to_do
    }


def save_dict_to_json(data: Dict[str, Any], file_path: Path) -> None:
    """
    Save a Python dictionary to a JSON file.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary to serialize into JSON.
    file_path : str
        Destination file path for the JSON output. Directories are created if necessary.

    """
    try:
        # Create directories if the path includes them
        dir_name = os.path.dirname(file_path)
        if dir_name:  # avoid errors if file_path is just "file.json"
            os.makedirs(dir_name, exist_ok=True)

        # Write the JSON file with UTF-8 support
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    except (TypeError, OSError) as e:
        raise ValueError("Failed to save dictionary to JSON.")


def to_jsonable(obj):
    if isinstance(obj, ndarray):
        return obj.tolist()
    if isinstance(obj, integer):
        return int(obj)
    if isinstance(obj, floating):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


def save_list_to_json(data_list, output_path):
    jsonable = to_jsonable(data_list)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(jsonable, f, indent=4, ensure_ascii=False)

def evaluate_save_run(base_result: Path, dt: LtrDataset, model: WrapperModel, training_time: float,
                      k_values: Optional[List[int]] = None, verbose: bool = False, save_stat: bool = True,
                      save_model: bool = False, prefix: Optional[str] = None) -> Dict:
    """

    Parameters
    ----------
    base_result : Path
        Base directory to save results/logs.

    dt : LtrDataset
        Dataset for evaluation.

    model : WrapperModel
        Result dictionary from training, containing at least keys "model" and "logs".

    training_time: float
        Time taken for training in seconds.

    k_values: Optional[List[int]], default None
        List of cutoff values for NDCG evaluation. If None, defaults to [5, 10, 15, 20, 25].
    verbose : bool, default False
        If True, print progress messages.

    save_model : bool, default False
        If True, save the trained model to disk.

    save_stat: bool, default True
        If True, save the training and evaluation statistics to disk.

    prefix : Optional[str], default None
        Optional prefix to add to log and model filenames (e.g., "best_"). If None, no prefix is added.
    Returns
    -------
    bool
        True if the run completed successfully.
    """
    assert base_result.exists(), "Base result directory does not exist."
    assert model.has_fitted, "Model has not been _fitted yet. Call fit() before evaluation."

    if k_values is None:
        k_values = [5, 10, 15, 20, 25]
    if prefix is None:
        prefix = ""
    # ---------------------------------------------------
    if verbose: print("Log ---", datetime.now(), "--- Start evaluation:", model.hash)
    start_evaluation = time.time()

    predictions = model.predict(dt)  # get predictions for evaluation dataset

    eval_result = {
        f"ndcg_{i}": evaluate(pred=predictions, labels=dt.y, groups_count=dt.group_count, k=i)
        for i in k_values
    }

    end_evaluation = time.time()
    if verbose: print("Log ---", datetime.now(), "--- End evaluation:", model.hash)
    # ---------------------------------------------------

    # mark the configuration parameter
    config = {(f"conf_{k}" if k != "hash" else k): v for k, v in model.config.items()}

    log = {
        **config,  # hyperparameters
        **model.get_logs(),  # training logs
        **eval_result,  # evaluation metrics

        "training_time": round(training_time, 4),
        "evaluation_time": round(end_evaluation - start_evaluation, 4)
    }

    if save_stat:
        if (base_result / f"log_{model.hash}.json").exists():
            raise ValueError("Log file already exists for this configuration.")
        save_dict_to_json(log, base_result / f"{prefix}log_{model.hash}.json")

    if save_model:
        model.save_model(base_result / f"{prefix}model_{model.hash}.pkl")

    if verbose: print("Log ---", datetime.now(), "--- Done!")
    return log


def train_with_config(base_result: Path, model: WrapperModel, train: LtrDataset, eval_dt: LtrDataset,
                      k_values: Optional[List[int]] = None, verbose: bool = False, save_model: bool = False,
                      save_stat: bool = True, prefix: Optional[str] = None) -> Dict:
    """
    Train and evaluate a single model configuration, logging results to JSON.

    Parameters
    ----------
    base_result : Path
        Base directory to save results/logs.

    model : WrapperModel
        WrapperModel instance to train and evaluate.

    train: LtrDataset
        Training dataset for model fitting.

    eval_dt : LtrDataset
        Validation dataset for evaluation.

    k_values : Optional[List[int]], default None
        List of cutoff values for NDCG evaluation. If None, defaults to [5, 10, 15, 20, 25].

    verbose : bool, default False
        If True, print progress messages.

    save_model : bool, default False
        If True, save the trained model to disk.

    save_stat: bool, default True
        If True, save the training and evaluation statistics to disk.

    prefix : Optional[str], default None
        Optional prefix to add to log and model filenames (e.g., "best_"). If None, no prefix is added.
    Returns
    -------
    bool
        True if the run completed successfully.

    """
    base_result.mkdir(exist_ok=True, parents=True)
    # ---------------------------------------------------

    if verbose: print("Log ---", datetime.now(), "--- Training with configuration:", model.hash)
    start_training = time.time()

    model.fit(t=train)  # train model

    end_training = time.time()
    if verbose: print("Log ---", datetime.now(), "--- End with configuration:", model.hash)
    # ---------------------------------------------------

    training_time = end_training - start_training
    return evaluate_save_run(base_result=base_result, dt=eval_dt, model=model, training_time=training_time,
                             k_values=k_values, verbose=verbose, save_stat=save_stat, save_model=save_model,
                             prefix=prefix)


def extract_best_confs(base_result: Path, new_index: Optional[List[int]], file_prefix: Optional[str] = None) -> Dict:
    """
    Parameters
    ----------
    base_result : Path
        Directory containing JSON files with evaluation results. Each JSON file should have keys starting with "conf
        " for configuration parameters and "ndcg_" for evaluation metrics.
    new_index : Optional[List[int]]
        New index to assign to the resulting DataFrame. Must match the number of unique configurations.
    file_prefix: Optional[str]
        If provided, only JSON files starting with this prefix will be considered.

    Returns
    -------
    Dict[int, Dict[str, Any]]
        A dictionary where keys are the new index values and values are dictionaries of configuration parameters
    """
    assert base_result.exists() and base_result.is_dir(), "base_result must be an existing directory."

    if file_prefix is None:
        file_prefix = ""

    grid_search_list = []
    for p in base_result.glob(f"{file_prefix}*.json"):
        if not p.is_file():
            continue
        with p.open("r") as ff:
            data = json.load(ff)
            grid_search_list.append(DataFrame.from_dict(data, orient="index").T)

    assert grid_search_list, "No JSON files found in base_result directory."

    grid_search_df: DataFrame = pd.concat(grid_search_list, axis=0, ignore_index=True)

    best_idx = (grid_search_df
                .filter(like="ndcg_")
                .map(lambda x: x[0])
                .apply(pd.to_numeric, errors="coerce")
                .idxmax(axis=0)
                )
    selected = grid_search_df.loc[best_idx].reset_index(drop=True)
    selected.index = list(new_index)

    selected = selected.filter(like="conf_")
    selected.columns = [col.replace("conf_", "") for col in selected.columns]

    return selected.to_dict(orient="index")
