# RuleTreeRank

<p align="center">
  <img src="imgs/logo.png" alt="RuleTreeRank logo" width="360">
</p>

RuleTreeRank (RTR) is an interpretable Learning-to-Rank framework that models ranking as a two-stage process: it first groups items with a shallow rule tree, then refines the score locally through instance-based comparisons.

RTR is designed for query-supported ranking problems. In candidate screening, for example, a query is a job offer and each item is a candidate. The same candidate can be relevant for one job and irrelevant for another, so the final ranking is induced within each query by sorting the RTR scores in descending order.

A visual explanation of the method is available in [imgs/explanation.pdf](imgs/explanation.pdf).

## What RTR Does

- **Stage I: rule-based grouping.** A shallow `RuleTreeRegressor` partitions the feature space into interpretable leaves and assigns each item a coarse score.
- **Stage II: local refinement.** Inside each leaf, RTR learns an interpretable pairwise distance model and uses a local k-NN correction over the Stage-I residuals.
- **Explanations.** The first stage gives a decision path, while the second stage exposes the local neighbours and pairwise rules used for the correction.

The package also exposes `MixedRTR`, a query-aware variant that trains the local refinement by both tree leaf and query when query-specific historical context is required.

## Installation

RTR currently ships as source code in this repository.

```bash
git clone <repository-url>
cd RuleTreeRank-DS2026
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Check that the local imports work:

```bash
python -c "from ruletreerank import RuleTreeRank; print('RTR import ok')"
```

## Minimal Usage

```python
import numpy as np
from RuleTree import RuleTreeRegressor

from ltr_utility import ModelParam
from ruletreerank import KNNRegFast, PairwiseDistanceTree, RuleTreeRank

X_train = np.random.default_rng(0).normal(size=(100, 5))
y_train = X_train @ np.array([1.4, -0.7, 0.5, 0.0, 0.2])
q_train = np.repeat(np.arange(10), 10)

model = RuleTreeRank(
    distance_f=ModelParam(PairwiseDistanceTree, {
        "base_regressor": ModelParam(RuleTreeRegressor, {"max_depth": 2, "random_state": 0}),
        "feature_concat": False,
        "feature_diff": True,
        "feature_sq_diff": True,
        "subsample": 0.5,
    }),
    aggregation_f=ModelParam(KNNRegFast, {"n_neighbors": 5, "n_jobs": 1}),
    base_regressor=RuleTreeRegressor(max_depth=2, random_state=0),
    dist_objective="residuals",
)

model.fit(X_train, y_train, q_train)

scores = model.predict(X_train, q=q_train, output="full")
ranking_for_query_0 = np.argsort(-scores[q_train == 0])
```

## Mathematical View

Ranking data is represented as a collection of queries and their items:

$$
\mathcal{D} = \{(q, X_q, y_q)\}_{q \in \mathcal{Q}},
\quad
X_q = \{\mathbf{x}_1,\ldots,\mathbf{x}_{n_q}\}.
$$

RTR learns a two-stage pointwise scoring function:

$$
f(\mathbf{x}) = r(\mathbf{x}) + s(\mathbf{x}).
$$

The first term, $r(\mathbf{x})$, is the score assigned by a shallow decision tree. If $\ell(\mathbf{x})$ is the leaf reached by item $\mathbf{x}$, the baseline score is the leaf mean:

$$
r(\mathbf{x}) =
\frac{1}{|\ell(\mathbf{x})|}
\sum_{\mathbf{x}_j \in \ell(\mathbf{x})} y_j.
$$

Training residuals are then computed as $\varepsilon_j = y_j - r(\mathbf{x}_j)$. The second term, $s(\mathbf{x})$, is a local correction: for a test item $\mathbf{x}^\star$, RTR retrieves the $k$ nearest training items in the same leaf using a learned pairwise distance model, then averages their residuals:

$$
s(\mathbf{x}^\star) =
\frac{1}{k}
\sum_{\mathbf{x}_j \in N_k(\mathbf{x}^\star)} \varepsilon_j.
$$

The final ranking for a query is obtained by sorting items by $f(\mathbf{x})$ in decreasing order.

## Examples

- [examples/min_example.ipynb](examples/min_example.ipynb): minimal RTR example on a scikit-learn dataset, with tables and plots showing the double-stage behaviour.
- [examples/synthetic_ranking_task.ipynb](examples/synthetic_ranking_task.ipynb): synthetic ranking task with a hidden linear scoring function and relevance labels induced by ranking.
- [imgs/explanation.pdf](imgs/explanation.pdf): compact visual explanation of the RTR pipeline.

## Repository Layout

- `ruletreerank/`: core RTR, MixedRTR, pairwise distance tree, and fast k-NN components.
- `ltr_utility/`: shared interfaces, dataset utilities, query splitting, clustering, and explanation helpers.
- `examples/`: executable notebooks for minimal and synthetic ranking workflows.
- `experiments/`: experiment scripts and evaluation notebooks.
