from typing import Tuple, Dict, Optional

import numpy as np
from numpy import ndarray
from sklearn.model_selection import train_test_split

from ltr_utility.dataset.ltr_dataset import LtrDataset


def process_split(dataset: Dict, split_key: str, num: int) -> Tuple[ndarray, ndarray, ndarray]:
    """
        Extracts, concatenates, and formats a specific split (e.g., 'train', 'test')
        from a nested dictionary dataset into flat NumPy arrays suitable for
        Learning-to-Rank algorithms.

        Parameters
        ----------
        dataset : Dict
            A dictionary containing the dataset. Expected to be indexed by query/group
            number, then by split key, returning a tuple of (features, labels).
        split_key : str
            The key indicating which data split to extract (e.g., 'train', 'vali', 'test').
        num : int
            The total number of queries/group to extract from the dataset.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            A tuple containing three 1D or 2D NumPy arrays:
            - x_final: Concatenated feature arrays.
            - y_final: Concatenated true label arrays.
            - q_final: Query IDs indicating which query each row belongs to.
    """
    x_list = [dataset[i][split_key][0] for i in range(num)]
    y_list = [dataset[i][split_key][1] for i in range(num)]

    lengths = [len(x) for x in x_list]

    x_final = np.concatenate(x_list, axis=0)
    y_final = np.concatenate(y_list, axis=0)
    q_final = np.repeat(np.arange(num), lengths)

    return x_final, y_final, q_final


def generate_complex_target1(X: ndarray, noise_std: float = 3, random_seed: int = 42) -> ndarray:
    """
        Generates a complex, continuous target variable by combining linear,
        interaction, trigonometric, and local (exponential) components based on
        the input features X.

        Parameters
        ----------
        X : ndarray
            2D array of input features with shape (n_samples, n_features).
        noise_std : float, optional
            Standard deviation of the Gaussian noise added to the final target.
        random_seed : int, optional
            Seed for the random number generator to ensure reproducibility.
            Default is 42.

        Returns
        -------
        ndarray
            1D array of generated target values with shape (n_samples,).
    """
    rng = np.random.default_rng(random_seed)
    n_samples, n_features = X.shape

    weights_linear = rng.uniform(-1, 1, n_features)
    y_linear = X @ weights_linear

    X_rolled = np.roll(X, shift=-1, axis=1)
    y_interaction = np.sum(X * X_rolled, axis=1)
    freq_weights = rng.uniform(0.5, 2.0, n_features)

    y_trig = np.sin(np.dot(X, freq_weights) * np.pi) + \
             np.cos(np.linalg.norm(X[:, :n_features // 2], axis=1) * 2)

    y_local = 3.0 * np.exp(-np.sum(X[:, :3] ** 2, axis=1))  # Uses first 3 features as trigger

    y_clean = (2.5 * y_linear) + (0.2 * y_interaction) + (0.5 * y_trig) + (0.1 * y_local)
    return y_clean + rng.normal(2, noise_std, n_samples)


def normalize(y: ndarray, num_bins: int = 4) -> ndarray:
    """
        Discretizes a continuous target variable into categorical relevance grades (bins).

        Parameters
        ----------
        y : np.ndarray
            1D array of continuous target values.
        num_bins : int, optional
            The number of bins (relevance grades) to divide the data into.
            Default is 4.

        Returns
        -------
        np.ndarray
            1D array of integer relevance grades ranging from 0 to (num_bins - 1).
    """
    labels = np.asarray([*range(num_bins)])

    min_val, max_val = np.min(y), np.max(y)
    edges = np.linspace(min_val, max_val, num_bins + 1)
    inner_edges = edges[1:-1]
    indices = np.digitize(y, inner_edges)

    return labels[indices]


def generate_synthetic_ltr(num_query: int, doc_x_query: int, features: int, x_mean: float = 0, x_std: float = 1,
                           num_bins: int = 4, random_seed: Optional[float] = None) -> LtrDataset:
    """
        Generates a global synthetic Learning-to-Rank dataset by creating all documents
        and queries at once, then sequentially splitting them into train, validation,
        and test sets.

        Parameters
        ----------
        num_query : int
            Total number of queries.
        doc_x_query : int
            Number of documents per query.
        features : int
            Number of features per document.
        x_mean : float, optional
            Mean of the normal distribution for feature generation.
        x_std : float, optional
            Standard deviation of the normal distribution for feature generation.
        num_bins : int, optional
            Number of relevance grades.
        random_seed : Optional[float], optional
            Random seed for reproducibility. If None, a random integer is used.

        Returns
        -------
        Tuple[LtrDataset, LtrDataset, LtrDataset]
            A tuple containing the training, validation, and test LtrDataset objects.
        """
    if random_seed is None:
        random_seed = np.random.randint(1, 10000)

    x = np.random.normal(x_mean, x_std, (num_query * doc_x_query, features))
    y_float = generate_complex_target1(x, random_seed=random_seed)
    y = normalize(y_float, num_bins)

    q = np.repeat([*range(0, num_query)], doc_x_query)

    return LtrDataset(x=x, y=y, q=q)


def generate_query_synthetic_ltr(num_query: int, doc_x_query: int, features: int, x_mean: float = 0,
                                 x_std: float = 1, num_bins: int = 4, train_size: float = .6,
                                 valid_size: float = .2, same_seed: bool = True,
                                 random_seed: Optional[float] = None) -> Tuple[LtrDataset, LtrDataset, LtrDataset]:
    """
        Generates a synthetic Learning-to-Rank dataset iteratively by query.
        Splits the documents *within* each query into train, validation, and test sets,
        then aggregates them.

        Parameters
        ----------
        num_query : int
            Total number of queries.
        doc_x_query : int
            Number of documents per query.
        features : int
            Number of features per document.
        x_mean : float, optional
            Mean of the normal distribution for feature generation.
        x_std : float, optional
            Standard deviation of the normal distribution for feature generation.
        num_bins : int
            Number of relevance grades.
        train_size : float, optional
            Proportion of documents *per query* in the train split.
        valid_size : float, optional
            Proportion of documents *per query* in the validation split.
        random_seed : Optional[float], optional
            Random seed for reproducibility. If None, a random integer is used.
        same_seed : bool, optional
            If True, uses the same random seed for the target generation of all queries.
            Default is True.

        Returns
        -------
        Tuple[LtrDataset, LtrDataset, LtrDataset]
            A tuple containing the aggregated training, validation, and test LtrDataset objects.
        """
    # Adjust the valid size to be a fraction of the remaining data after training split
    valid_size = valid_size / (1.0 - train_size)

    query_dataset = {}

    if same_seed:
        r = random_seed if random_seed is not None else np.random.randint(1, 10000)
        seed = np.repeat(r, num_query)
    else:
        seed = np.random.randint(1, 10000, size=num_query)

    for i in range(num_query):
        x = np.random.normal(x_mean, x_std, (doc_x_query, features))
        y = normalize(y=generate_complex_target1(x, random_seed=seed[i]),
                      num_bins=num_bins)

        train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=train_size)
        valid_x, test_x, valid_y, test_y = train_test_split(test_x, test_y, train_size=valid_size)

        query_dataset[i] = {
            "train": (train_x, train_y),
            "valid": (valid_x, valid_y),
            "test": (test_x, test_y)
        }
    train_x, train_y, q_train = process_split(query_dataset, "train", num_query)
    valid_x, valid_y, q_valid = process_split(query_dataset, "valid", num_query)
    test_x, test_y, q_test = process_split(query_dataset, "test", num_query)

    train = LtrDataset(x=train_x, y=train_y, q=q_train)
    valid = LtrDataset(x=valid_x, y=valid_y, q=q_valid)
    test = LtrDataset(x=test_x, y=test_y, q=q_test)

    return train, valid, test


class RandomPredictor:
    """
    A baseline model that outputs random Gaussian noise as predictions.
    Useful for testing evaluation pipelines or acting as a lowest-bound baseline.
    """

    def __init__(self, random_seed: Optional[int] = 42):
        if random_seed is None:
            random_seed = np.random.randint(1, 10000)
        self.rng = np.random.default_rng(random_seed)

    def predict(self, X: np.ndarray, *args) -> np.ndarray:
        """
        Generates random predictions for the given input features.

        Parameters
        ----------
        X : np.ndarray
            2D array of input features

        Returns
        -------
        np.ndarray
            1D array of random normal values.
        """
        return self.rng.normal(0, 1, X.shape[0])
