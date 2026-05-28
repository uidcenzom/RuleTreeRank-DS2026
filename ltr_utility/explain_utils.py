from typing import List, Dict, Any

from numpy import ndarray

from ruletreerank import PairwiseDistanceTree
from . import TreeModel


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


def get_pdt_rules_list(x: ndarray, Z: ndarray, model: PairwiseDistanceTree) -> List:
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
