from typing import List, Dict, Any, Tuple, TYPE_CHECKING

import numpy as np
from numpy import ndarray
from pandas import DataFrame

from . import TreeModel

if TYPE_CHECKING:  # ruletreerank imports this module, so keep the cycle out of runtime
    from ruletreerank import PairwiseDistanceTree


def list_roles(path: List, rule_dict: Dict[str, Any], list_rules: List[str] = None) -> List:
    """
    Traverse a tree rule dictionary along a path and collect textual rules.

    Parameters
    ----------
    path : list
        Sequence of left/right path markers after the root.
    rule_dict : dict
        Nested rule dictionary returned by a RuleTree-style model.
    list_rules : list, optional
        Accumulator used during recursive traversal.
    """
    if list_rules is None: list_rules = []
    if rule_dict["is_leaf"]: return list_rules[::-1]
    return list_roles(
        path=path[1:],
        rule_dict=rule_dict["left_node" if path[0] == "l" else "right_node"],
        list_rules=[rule_dict["textual_rule"], *list_rules]
    )


def get_rules_list(x: ndarray, model: TreeModel) -> List:
    """
    Return the shallow-tree rules followed by one input instance.

    Parameters
    ----------
    x : np.ndarray
        Instance to explain.
    model : TreeModel
        Fitted tree model exposing `apply` and `get_rules`.
    """
    if x.ndim == 1: x = x.reshape(-1, 1)
    return list_roles(path=list(model.apply(x)[0])[1:], rule_dict=model.get_rules())


def get_pdt_rules_list(x: ndarray, Z: ndarray, model: "PairwiseDistanceTree") -> List:
    """
    Return PairwiseDistanceTree rules for pairs between `x` and each row in `Z`.

    Parameters
    ----------
    x : np.ndarray
        Query instance being explained.
    Z : np.ndarray
        Neighbor instances compared against `x`.
    model : PairwiseDistanceTree
        Fitted pairwise distance tree.
    """
    if x.ndim == 1: x = x.reshape(-1, 1)
    assert Z.ndim == 2, x.shape[1] == Z.shape[1]

    result = list(map(
        lambda z: list_roles(path=list(model.apply(x, z.reshape(1, -1))[0])[1:], rule_dict=model.get_rules()),
        Z
    ))
    return result


def leaf_diagnostics(model, X, y, target_name: str = "target") -> Tuple[DataFrame, DataFrame]:
    """
    Summarise the two RuleTreeRank stages leaf by leaf.

    Stage 1 assigns a coarse score per leaf; stage 2 fits a local k-NN model on the
    residuals inside that leaf. This pairs the residual distribution with the local
    model that was fitted on it.

    Parameters
    ----------
    model : RuleTreeRank
        Fitted ranker.
    X : pd.DataFrame or np.ndarray
        Training rows the model was fitted on.
    y : np.ndarray
        Targets aligned with `X`.
    target_name : str
        Column name to use for the target, e.g. "target" or "relevance".

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        The row-level diagnostic frame and the per-leaf summary.
    """
    X = np.asarray(X)
    leaf = model._shallow_dt.apply(X)
    stage1 = model._shallow_dt.predict(X)

    per_row = DataFrame({
        "leaf": leaf,
        target_name: y,
        "stage1_score": stage1,
        "residual_after_stage1": y - stage1,
    })

    residuals = per_row.groupby("leaf").agg(
        n_train=(target_name, "size"),
        **{f"{target_name}_mean": (target_name, "mean")},
        stage1_mean=("stage1_score", "mean"),
        residual_mean=("residual_after_stage1", "mean"),
        residual_std=("residual_after_stage1", "std"),
    )

    local_models = DataFrame([
        {
            "leaf": leaf_id,
            "knn_rows": int(agg._fit_X.shape[0]),
            "n_neighbors": int(agg.n_neighbors),
            "learned_distance": agg.custom_metric_func is not None,
        }
        for leaf_id, agg in model._leaf_dist_map.items()
    ]).set_index("leaf")

    per_leaf = residuals.join(local_models).sort_values("n_train", ascending=False)
    return per_row, per_leaf
