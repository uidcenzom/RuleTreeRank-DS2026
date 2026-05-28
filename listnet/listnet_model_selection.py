import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd
from tqdm import tqdm

from listnet.ranklib_manager import RankLibTrainer, is_recoverable_listnet_error, zero_ndcg_scores
from ltr_utility.dataset.ltr_dataset import LtrDataset


def listnet_model_selection(
        cache_path: Path, rankLib_path: Path, query_groups: Dict[Any, Iterable[Any]],
        train: LtrDataset, valid: LtrDataset, test: LtrDataset, lr_values:List[float],
        epoch_values: List[int], metric: str = "NDCG@10", output_path: Optional[Path] = None,
        ) -> List[Dict[str, Any]]:
    """
    Model selection per ListNet su griglia di learning rate ed epoch.

    Per ogni query prova tutte le coppie `(lr, epoch)`, seleziona solo il
    risultato con `ndcg_valid` migliore e salva quei best result in JSON.
    """

    cache_path.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = cache_path / "listnet_best_lr_epoch_model_selection.json"

    results: List[Dict[str, Any]] = []
    hyperparams = list(product(lr_values, epoch_values))

    for qxm, q_ids_iter in query_groups.items():
        q_ids = list(q_ids_iter)
        best_for_group = []

        for q_id in tqdm(q_ids, desc=f"ListNet model selection qxm={qxm}"):
            tr_path = cache_path / f"sub_train.txt"
            vl_path = cache_path / f"sub_valid.txt"
            ts_path = cache_path / f"sub_test.txt"

            try:
                train[[q_id]].to_ltr(tr_path)
                valid[[q_id]].to_ltr(vl_path)
                test[[q_id]].to_ltr(ts_path)

                best_result: Optional[Dict[str, Any]] = None

                for lr, epoch in hyperparams:
                    try:
                        scores = RankLibTrainer(jar_path=rankLib_path, algorithm="ListNet", metric=metric,
                            epoch=epoch, lr=lr, silent=True).fit_predict(train_file=tr_path, validation_file=vl_path,
                            test_file=ts_path)
                    except RuntimeError as exc:
                        if not is_recoverable_listnet_error(exc):
                            raise
                        scores = zero_ndcg_scores()
                    candidate = {
                        "q_id": _jsonable(q_id),"best_lr": float(lr), "best_epoch": int(epoch),
                        **scores,
                    }
                    if best_result is None or candidate["ndcg_valid"] > best_result["ndcg_valid"]:
                        best_result = candidate

                if best_result is not None: best_for_group.append(best_result)
            finally:
                for path in (tr_path, vl_path, ts_path): path.unlink(missing_ok=True)

        results.append(
            {"qxm": _jsonable(qxm),"q_ids": [_jsonable(q_id) for q_id in q_ids], "best_results": best_for_group}
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=4), encoding="utf-8")
    return results

def _jsonable(value: Any) -> Union[int, float, str, bool, None]:
    if hasattr(value, "item"):
        return value.item()
    return value


def _safe_name(value: Any) -> str:
    return str(_jsonable(value)).replace("/", "_").replace(" ", "_")


def make_listnet_result(results: List):
    fold_map = {"ndcg_training": "train", "ndcg_valid": "valid", "ndcg_test": "test", }
    df = pd.DataFrame(results).explode(["q_ids", "best_results"], ignore_index=True)

    return (
        df[["qxm", "q_ids"]]
        .join(pd.json_normalize(df["best_results"])[list(fold_map)])
        .melt(["qxm", "q_ids"], var_name="fold", value_name="ndcg")
        .assign(fold=lambda d: d["fold"].map(fold_map))
        .groupby(["qxm", "fold"])["ndcg"]
        .agg(mean="mean", median="median", std="std")
        .reset_index()
        .assign(model="ListNET", k=10)
        [["model", "fold", "qxm", "k", "mean", "std", "median"]]
    )
