from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def load_result_file(dataset_name: str, dataset_dir: str, model_name: str, filename: str) -> pd.DataFrame:
    """Load one model-selection result file and aggregate validation NDCG by configuration.

    The input JSON is expected to contain one row per query, with parallel
    `configs` and `results` lists. Each element of `configs` describes a
    hyperparameter/configuration candidate, and the matching element of
    `results` contains that configuration's validation NDCG for the query.

    The function explodes those lists to query-configuration rows, normalizes
    the configuration dictionaries into columns, and then aggregates back to
    one row per `dataset x model x configuration`.

    Parameters
    ----------
    dataset_name:
        Human-readable dataset label to store in the returned dataframe, e.g.
        `"MQ2007"` or `"FINDHR"`.
    dataset_dir:
        Directory name under `../query_based` containing the `results/` folder.
        This function assumes it is run from `experiments/Sensitivity` or from
        another working directory where `../query_based/<dataset_dir>/results`
        resolves correctly.
    model_name:
        Human-readable model label to store in the returned dataframe, e.g.
        `"Mix-RuleTreeRank"`.
    filename:
        Result filename inside the dataset's `results/` directory, e.g.
        `"mix_rtr_query_model_selection.json"`.

    Returns
    -------
    pd.DataFrame
        One row per `dataset x model x configuration`, with all configuration
        columns plus:

        - `ndcg`: mean validation NDCG over queries.
        - `ndcg_std`: standard deviation of validation NDCG over queries.
        - `n_queries`: number of query-level observations used.
    """
    data = pd.read_json(Path("../query_based") / dataset_dir / "results" / filename)

    long = (data[["query", "configs", "results"]]
            .explode(["configs", "results"])
            .reset_index(names="query_idx")
            .rename(columns={"results": "ndcg"})
            )
    configs = pd.json_normalize(long["configs"])
    config_cols = configs.columns.tolist()

    result = pd.concat([
        long.drop(columns="configs").reset_index(drop=True),
        configs.reset_index(drop=True),
    ], axis=1)

    result["dataset"] = dataset_name
    result["model"] = model_name
    result["ndcg"] = pd.to_numeric(result["ndcg"], errors="coerce")

    return (
        result.groupby(["dataset", "model"] + config_cols, dropna=False)["ndcg"]
        .agg(ndcg="mean", ndcg_std="std", n_queries="count")
        .reset_index()
    )


def load_all_results(datasets: dict[str, str], models: dict[str, str]) -> pd.DataFrame:
    """Load and concatenate all dataset-model result files.

    This is the main entry point for building the sensitivity-analysis table.
    It calls `load_result_file` for every pair in `datasets x models`, then
    appends two context-normalized columns:

    - `ndcg_centered`: configuration NDCG minus the mean NDCG within the same
      `dataset x model` block.
    - `regret`: configuration NDCG minus the best NDCG within the same
      `dataset x model` block. This is zero for the best configuration and
      negative for all others.

    Parameters
    ----------
    datasets:
        Mapping from display dataset name to directory name. Example:
        `{"MQ2007": "MQ2007", "YAHOO": "YAHOO"}`.
    models:
        Mapping from display model name to result filename. Example:
        `{"Mix-RuleTreeRank": "mix_rtr_query_model_selection.json"}`.

    Returns
    -------
    pd.DataFrame
        Concatenated configuration-level results across all requested datasets
        and models.

    """
    rows = []
    for dataset_name, dataset_dir in datasets.items():
        for model_name, filename in models.items():
            rows.append(load_result_file(dataset_name, dataset_dir, model_name, filename))

    result = pd.concat(rows, ignore_index=True)

    result["ndcg_centered"] = result["ndcg"] - result.groupby(["dataset", "model"])["ndcg"].transform("mean")
    result["regret"] = result["ndcg"] - result.groupby(["dataset", "model"])["ndcg"].transform("max")

    return result


def as_tuple(value):
    """Return `value` as a tuple.

    Pandas returns a scalar group key when grouping by a single column and a
    tuple group key when grouping by multiple columns. This helper normalizes
    both cases so code can safely zip group keys with column names.
    """
    return value if isinstance(value, tuple) else (value,)


def ordered_levels(values):
    """Return feature levels in a deterministic low-to-high order.

    The ordering is used to compute signed effects. For example, if a feature
    is boolean, the directional effect is computed as `True - False`; if a
    feature is numeric, it is computed as `max_value - min_value`.

    Categorical string features are sorted lexicographically. For those
    features, the sign is still deterministic, but it may not have a natural
    semantic meaning unless the category order is meaningful for the analysis.

    Parameters
    ----------
    values:
        Iterable of observed feature levels.

    Returns
    -------
    list
        Feature levels ordered from low/reference level to high/comparison
        level.

    Examples
    --------
    >>> ordered_levels([True, False])
    [False, True]
    >>> ordered_levels([8, 2, 4])
    [2, 4, 8]
    >>> ordered_levels(["y", "dist"])
    ['dist', 'y']
    """
    levels = pd.Series(list(values)).dropna().unique().tolist()

    if len(levels) == 0:
        return []

    if all(isinstance(value, (bool, np.bool_)) for value in levels):
        return [value for value in [False, True] if value in levels]

    if all(isinstance(value, (int, float, np.integer, np.floating)) for value in levels):
        return sorted(levels)

    return sorted(levels, key=str)


def summarise_by_blocks(rows: pd.DataFrame, item_cols:List=None, value_col: str="effect_range",
                        block_cols:List=None) -> pd.DataFrame:
    """Summarize matched-effect rows while giving each block equal weight.

    Sensitivity rows can be numerous for some datasets or models because they
    may contain more matched configurations. A raw average over all rows would
    let larger blocks dominate the final estimate. This function avoids that
    by using a two-stage aggregation:

    1. Average `value_col` within each `item_cols x block_cols` group.
    2. Average those block-level values across blocks for each item.

    In this project, typical blocks are:

    - `["dataset"]` for the primary analysis on one model.
    - `["dataset", "model"]` for robustness across model variants.

    Parameters
    ----------
    rows:
        Dataframe containing matched-effect rows, such as the output of
        `matched_main_effects`.
    item_cols:
        Columns identifying the entity being summarized, e.g. `["feature"]` or
        `["feature_1", "feature_2", "pair"]`.
    value_col:
        Numeric column to summarize, e.g. `"effect_range"`.
    block_cols:
        Columns defining equal-weight blocks, e.g. `["dataset"]`.

    Returns
    -------
    pd.DataFrame
        Summary dataframe sorted by decreasing `mean`, with columns:

        - `mean`, `median`, `std`: block-level summary statistics.
        - `se`: standard error across blocks.
        - `ci95`: approximate 95% confidence interval width, computed as
          `1.96 * se`.
        - `n_blocks`: number of blocks used.
        - `n_matches`: number of raw matched rows before block aggregation.

    """
    if item_cols is None: item_cols = ["feature"]
    if block_cols is None: block_cols = ["dataset", "model"]

    item_cols, block_cols = list(item_cols), list(block_cols)

    if rows.empty:
        columns = item_cols + ["mean", "median", "std", "se", "ci95", "n_blocks", "n_matches"]
        return pd.DataFrame(columns=columns)

    block_values = (
        rows.groupby(item_cols + block_cols, dropna=False)[value_col]
        .mean()
        .reset_index()
    )

    summary = (
        block_values.groupby(item_cols, dropna=False)[value_col]
        .agg(mean="mean", median="median", std="std", n_blocks="count")
        .reset_index()
    )

    n_matches = rows.groupby(item_cols, dropna=False).size().rename("n_matches").reset_index()
    summary = summary.merge(n_matches, on=item_cols, how="left")
    summary["std"] = summary["std"].fillna(0)
    summary["se"] = summary["std"] / np.sqrt(summary["n_blocks"].clip(lower=1))
    summary["ci95"] = 1.96 * summary["se"]

    return summary.sort_values("mean", ascending=False).reset_index(drop=True)


def matched_main_effects(df: pd.DataFrame, features: list[str], context_cols: list[str] = None):
    """Compute matched main effects for each feature.

    For each feature, the function holds all other features fixed and compares
    the observed NDCG values across the levels of the target feature. The main
    effect for one matched group is:

    `effect_range = max(ndcg across feature levels) - min(ndcg across feature levels)`

    This is an unsigned impact measure. It tells how much performance changes
    when only the target feature varies, but it does not encode which level is
    better.

    The function also computes a signed directional effect:

    `directional_effect = ndcg(high_level) - ndcg(low_level)`

    Interpretation:

    - `directional_effect > 0`: increasing/enabling the feature improves NDCG
      on average.
    - `directional_effect < 0`: increasing/enabling the feature hurts NDCG on
      average.
    - `directional_effect ~= 0`: no clear signed effect.

    For boolean features, `low_level=False` and `high_level=True`. For numeric
    features, low/high are min/max. For string categorical features, low/high
    are lexicographic, so inspect the category meaning before interpreting the
    sign.

    Parameters
    ----------
    df:
        Configuration-level dataframe produced by `load_all_results` or
        `load_result_file`. It must contain `ndcg`, the columns in `features`,
        and the columns in `context_cols`.
    features:
        Feature/configuration columns to analyze. For a target feature, all
        other columns in this list are treated as matching controls.
    context_cols:
        Columns that define separate analysis contexts. The default compares
        configurations only within the same `dataset x model`.

    Returns
    -------
    pd.DataFrame
        Feature-level summary. The original absolute-impact columns are kept:
        `mean`, `median`, `std`, `se`, `ci95`, `n_blocks`, and `n_matches`.
        These summarize `effect_range`.

        Additional signed-effect columns are added:

        - `mean_direction`: average signed `high_level - low_level` effect.
        - `median_direction`: median signed effect across blocks.
        - `ci95_direction`: approximate 95% confidence interval width for the
          signed effect.
        - `n_direction_blocks`: number of blocks used for the signed summary.
        - `n_direction_matches`: number of matched comparisons used for the
          signed summary.

    For boolean features, `directional_effect` is the mean score difference
    between the enabled and disabled levels inside each matched block. For
    numeric or categorical features, it is the score difference between the
    ordered high and low levels.
    """
    if context_cols is None: context_cols = ["dataset", "model"]
    rows = []
    for feature in features:
        other_features = [candidate for candidate in features if candidate != feature]
        group_cols = context_cols + other_features

        grouped = (
            df.groupby(group_cols + [feature], dropna=False)["ndcg"]
            .mean()
            .reset_index()
        )

        for group_key, block in grouped.groupby(group_cols, dropna=False):
            values = block.groupby(feature, dropna=False)["ndcg"].mean().dropna()

            if len(values) < 2: continue
            levels = ordered_levels(values.index)
            low_level = levels[0] if len(levels) >= 1 else np.nan
            high_level = levels[-1] if len(levels) >= 2 else np.nan
            directional_effect = np.nan

            if len(levels) >= 2 and low_level in values.index and high_level in values.index:
                directional_effect = values.loc[high_level] - values.loc[low_level]

            row = dict(zip(group_cols, as_tuple(group_key)))
            row.update({
                "feature": feature,
                "effect_range": values.max() - values.min(),
                "directional_effect": directional_effect,
                "low_level": low_level,
                "high_level": high_level,
                "best_level": values.idxmax(),
                "worst_level": values.idxmin(),
                "n_levels": len(values),
            })
            rows.append(row)

    effect_rows = pd.DataFrame(rows)

    if effect_rows.empty:
        return summarise_by_blocks(effect_rows, block_cols=context_cols)

    impact = summarise_by_blocks(effect_rows, block_cols=context_cols)

    directional_rows = effect_rows.dropna(subset=["directional_effect"])
    if directional_rows.empty:
        return impact

    direction = summarise_by_blocks(
        directional_rows,
        value_col="directional_effect",
        block_cols=context_cols,
    ).rename(columns={
        "mean": "mean_direction",
        "median": "median_direction",
        "std": "std_direction",
        "se": "se_direction",
        "ci95": "ci95_direction",
        "n_blocks": "n_direction_blocks",
        "n_matches": "n_direction_matches",
    })

    direction_cols = [
        "feature",
        "mean_direction",
        "median_direction",
        "std_direction",
        "se_direction",
        "ci95_direction",
        "n_direction_blocks",
        "n_direction_matches",
    ]

    return impact.merge(direction[direction_cols], on="feature", how="left")
