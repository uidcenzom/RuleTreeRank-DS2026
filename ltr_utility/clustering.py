from enum import Enum
from typing import Optional, Union, List, Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

from ltr_utility.dataset.ltr_dataset import LtrDataset


class AggrMode(Enum):
    """Available strategies for reducing each query to one embedding."""

    FEATURE = 0


class QueryPartition:
    """
    Build fixed-size groups of similar queries from query-level embeddings.

    The current implementation aggregates feature distributions per query and
    groups the resulting embeddings with a nearest-neighbor cosine heuristic.
    """

    def __init__(self, n_elements: int, agg_mode: AggrMode, random_state: Optional[int] = None,
                 features: Optional[Union[slice, List[int]]] = None):
        """
        Parameters
        ----------
        n_elements : int
            Target number of queries or rows per group.
        agg_mode : AggrMode
            Aggregation strategy used before grouping.
        random_state : int, optional
            Seed used by visualization routines.
        features : slice or list, optional
            Feature subset used when aggregating an `LtrDataset`.
        """

        self.n_elements = n_elements
        self.aggr_mode = agg_mode
        self.random_state = random_state
        self.features = features

        self.model_ = None
        self.clusters_ = None
        self.embeddings_ = None

    def _aggregation(self, dt: LtrDataset) -> ndarray:
        """Convert an `LtrDataset` into one embedding per query."""
        match self.aggr_mode:
            case AggrMode.FEATURE:
                return self._feature_agg(dt)
            case _:
                raise ValueError("Unsupported aggregation _mode.")

    @staticmethod
    def _feature_agg(dt: LtrDataset) -> ndarray:
        """
        Aggregate every query by feature percentiles, mean, and median.

        Returns
        -------
        np.ndarray
            Matrix with shape `(n_queries, 4 * n_features)`.
        """
        return np.asarray([
            np.concatenate(
                [np.percentile(x, 25, axis=0), x.mean(axis=0), np.median(x, axis=0),
                 np.percentile(x, 75, axis=0)]
            )
            for _, x, _ in dt.iterate_queries()
        ])

    @staticmethod
    def create_uniform_groups(embeddings: ndarray, elements: int, metric: str):
        """
        Greedily partition embeddings into groups of nearby rows.

        Parameters
        ----------
        embeddings : np.ndarray
            Row-wise embeddings to group.
        elements : int
            Target group size.
        metric : str
            Distance metric passed to `sklearn.metrics.pairwise_distances`.
        """
        n_samples = embeddings.shape[0]

        assert elements > 0, "emb_x_group deve essere maggiore di 0"

        if elements > n_samples:  return [list(range(n_samples))]

        dist_matrix = pairwise_distances(embeddings, metric=metric)
        unassigned = set(range(n_samples))
        groups = []

        while len(unassigned) >= elements:
            current_seed = next(iter(unassigned))
            distances_from_seed = dist_matrix[current_seed]
            valid_candidates = [(idx, distances_from_seed[idx]) for idx in unassigned]
            valid_candidates.sort(key=lambda x: x[1])

            group_indices = [candidate[0] for candidate in valid_candidates[:elements]]
            group_indices.sort()
            groups.append(group_indices)

            unassigned.difference_update(group_indices)

        if unassigned: groups.append(list(unassigned))

        return groups

    def fit_cosine(self, dt: Union[LtrDataset, ndarray], aggregated: bool = False) -> dict:
        """
        Compute cosine-based query groups.

        Parameters
        ----------
        dt : LtrDataset or np.ndarray
            Dataset to aggregate, or precomputed row embeddings when `aggregated=True`.
        aggregated : bool
            If True, `dt` is interpreted as already aggregated embeddings.
        """

        if not aggregated:
            assert isinstance(dt, LtrDataset), "if it is not aggregated must be a LtRDataset"
            if self.features is not None:
                dt = dt.filter_columns(columns=self.features, inplace=False)

            embeddings = self._aggregation(dt)  # dim n_queries x m
        else:
            assert isinstance(dt, ndarray), "in aggregated dt must be ndarray"
            embeddings = dt

        pairs = self.create_uniform_groups(embeddings, self.n_elements, "cosine")

        if isinstance(dt, LtrDataset):
            pairs = map(lambda a: [int(dt.unique_q[i]) for i in a], pairs)

        return {i: c for i, c in enumerate(pairs)}

    def plot_tsne(self, dt: Union[LtrDataset, np.ndarray], clusters: Optional[Dict[int, Iterable[int]]] = None,
                  aggregated: bool = False, perplexity: float = 30.0, figsize=(8, 6),
                  title: str = "Query clustering (t-SNE)"):
        """
        Visualize query embeddings or precomputed embeddings with t-SNE.

        Parameters
        ----------
        clusters : dict, optional
            Optional cluster mapping used to color the projected points.
        aggregated : bool
            If True, `dt` is treated as an embedding matrix.
        """

        if not aggregated:
            assert isinstance(dt, LtrDataset)
            if self.features is not None:
                dt = dt.filter_columns(columns=self.features, inplace=False)
            embeddings = self._aggregation(dt)
        else:
            assert isinstance(dt, np.ndarray)
            embeddings = dt

        self.embeddings_ = embeddings

        labels = np.zeros(len(embeddings), dtype=int)

        if clusters is not None:
            self.clusters_ = clusters

            if isinstance(dt, LtrDataset):
                qid_to_pos = {int(qid): i for i, qid in enumerate(dt.unique_q)}
                for cid, qids in clusters.items():
                    for qid in qids:
                        if qid in qid_to_pos: labels[qid_to_pos[qid]] = cid
            else:
                for cid, indices in clusters.items():
                    for idx in indices: labels[idx] = cid
        else: self.clusters_ = None


        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=self.random_state, init="pca")
        tsne_emb = tsne.fit_transform(embeddings)
        self.tsne_ = tsne_emb

        # 4. Plot
        plt.figure(figsize=figsize)
        scatter = plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=labels, cmap="tab20", s=40, alpha=0.8)

        plt.title(title)
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")

        if clusters is not None:
            plt.legend(
                *scatter.legend_elements(),
                title="Cluster",
                loc="best",
                fontsize="small"
            )

        plt.tight_layout()
        plt.show()
