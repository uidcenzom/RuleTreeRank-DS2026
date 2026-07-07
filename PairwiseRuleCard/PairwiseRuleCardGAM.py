import copy

import numpy as np
from RuleTree import RuleTreeRegressor
try:
    # Only needed when fast != None (interaction detection). Optional heavy dependency.
    from interpret.utils import measure_interactions
except Exception:  # pragma: no cover - optional dependency
    measure_interactions = None
from joblib import Parallel, delayed
from ordered_set import OrderedSet
from scipy.spatial.distance import cosine
from tqdm.auto import tqdm

try:
    # Only needed for importance-based subsampling strategies (catboost/xgboost). Optional.
    from PairwiseRuleCard.feat_importance_wrapper import *
except Exception:  # pragma: no cover - optional dependency
    pass
from PairwiseRuleCard.utils import generate_pairwise_dataset


class PairwiseRuleCardGAM:
    def __init__(self, learning_rate=.1, patience=5, val_size=.15, base_estimator=RuleTreeRegressor(max_depth=3),
                 max_n_iter=100,

                 enforce_simmetry=False,

                 reuse_features=True,

                 base_pred='mean',  # mean or zero

                 fast='all',  # None, same_feature, all
                 use_pairwise=True,
                 use_difference=False,

                 metric='euclidean',

                 subsample=1.,
                 subsample_strategy='random',

                 score_metric='mae',  # mae or cosine

                 n_jobs=1,
                 random_state=42, verbose=False):
        if subsample_strategy is None:
            subsample = 1.

        self.learning_rate = learning_rate
        self.patience = patience
        self.val_size = val_size
        self.base_estimator = base_estimator
        self.max_n_iter = max_n_iter

        self.enforce_simmetry = enforce_simmetry

        self.reuse_features = reuse_features

        self.base_pred = base_pred

        self.fast = fast
        self.use_pairwise = use_pairwise
        self.use_difference = use_difference

        self.metric = metric

        self.subsample_strategy = subsample_strategy
        self.subsample = subsample

        self.score_metric = score_metric.lower()

        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.base_estimator.random_state = self.random_state

        self.feature_combinations_ = OrderedSet()
        self.feature_importance_ = None
        self.used_features_ = OrderedSet()

        if not use_pairwise and fast == 'same_features':
            raise ValueError('use_pairwise=False cannot be used with fast=same_features')

        if not use_pairwise and not use_difference:
            raise ValueError('At least one of use_pairwise or use_difference must be True')

        if self.score_metric not in {'mae', 'cosine'}:
            raise ValueError("score_metric must be one of {'mae', 'cosine'}.")

        np.random.seed(self.random_state)

    def _fast(self, X, y, admissible_features):
        if len(admissible_features) < 2:
            return OrderedSet()

        interactions = None
        if self.fast == 'same_features':
            interactions = [(i, i + self.original_n_features) for i in range(self.original_n_features)]

        return OrderedSet([c for c, _ in measure_interactions(X, y, objective='rmse', interactions=interactions)][:X.shape[0]])

    def _get_combinations(self, X, y):
        if self.feature_combinations_ is not None:
            self.feature_combinations_ = OrderedSet([(x,) for x in range(X.shape[1])]) - self.used_features_

        if self.fast is not None:
            self.feature_combinations_ |= self._fast(X, y, self.feature_combinations_)

        return self.feature_combinations_ - self.used_features_

    def predict(self, Xa: np.ndarray, Xb: np.ndarray):
        """Predict pairwise distances between aligned rows of Xa and Xb."""
        Xa = np.asarray(Xa)
        Xb = np.asarray(Xb)

        if Xa.shape != Xb.shape:
            raise ValueError("Xa and Xb must have same number of instances and features.")

        X_pairs = np.hstack([Xa, Xb])
        if self.use_difference:
            X_diff = np.abs(Xa - Xb)
            X = np.hstack([X_pairs, X_diff]) if self.use_pairwise else X_diff
        else:
            X = X_pairs

        prediction = np.ones((X.shape[0],)) * self.base_prediction_
        for feat_idx, est in self.estimators_:
            leafs = est.predict(X[:, feat_idx].reshape(X.shape[0], -1))
            prediction += self.learning_rate * leafs
        return prediction + np.abs(np.min(prediction))

    def fit(self, X: np.ndarray, y=None, mode="distance", model=None, pair_targets=None):
        """
        Fit the additive pairwise model.

        Parameters
        ----------
        X : np.ndarray
            Original feature matrix with shape (n_samples, n_features), or pre-built pairwise matrix when
            mode in {'distance', 'self_supervised'} and y is provided.
        y : np.ndarray or None
            Used according to mode:
            - 'supervised': class labels for X.
            - 'distance'/'self_supervised': if provided, interpreted as precomputed pairwise targets and X as X_pairs.
            - legacy: second matrix Xb when calling fit(Xa, Xb, ...).
        mode : str
            One of {'distance', 'supervised', 'model', 'precomputed', 'self_supervised'}.
        model : object or None
            Pretrained model exposing predict(X), used in mode='model'.
        pair_targets : np.ndarray or None
            Precomputed pairwise targets used in mode='precomputed', or optionally in 'distance'/'self_supervised'.
        """
        # Reset per-fit state so multiple fit calls on the same instance are independent.
        self.used_features_ = OrderedSet()
        self.feature_combinations_ = OrderedSet()

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError('X must be a 2D array.')

        legacy_Xb = None
        legacy_yb = None

        # Legacy dispatch: fit(Xa, Xb, ya=None, yb=None).
        if not isinstance(mode, str):
            legacy_Xb = np.asarray(y)
            y = None if mode is None else np.asarray(mode)
            legacy_yb = None if model is None else np.asarray(model)
            mode = 'distance'
            model = None
            pair_targets = None
        elif (
            isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] == X.shape[1]
            and mode in {'distance', 'self_supervised'} and model is None and pair_targets is None
        ):
            # Legacy shorthand: fit(Xa, Xb).
            legacy_Xb = np.asarray(y)
            y = None

        mode = mode.lower()
        allowed_modes = {'distance', 'supervised', 'model', 'precomputed', 'self_supervised'}
        if mode not in allowed_modes:
            raise ValueError(f"Invalid mode '{mode}'. Allowed values are {sorted(allowed_modes)}.")

        if legacy_Xb is not None:
            if legacy_Xb.ndim != 2:
                raise ValueError('Legacy Xb must be a 2D array.')
            if X.shape[1] != legacy_Xb.shape[1]:
                raise ValueError('Xa and Xb must have same number of features.')
            self.original_n_features = X.shape[1]
        elif mode in {'distance', 'self_supervised'} and y is not None and pair_targets is None and model is None:
            if X.shape[1] % 2 == 0:
                self.original_n_features = X.shape[1] // 2
            else:
                self.original_n_features = X.shape[1]
        else:
            self.original_n_features = X.shape[1]

        X_pairs, y_pairs = _gen_pairs(
            X=X,
            y=y,
            mode=mode,
            model=model,
            pair_targets=pair_targets,
            subsample_strategy=self.subsample_strategy,
            subsample=self.subsample,
            use_pairwise=self.use_pairwise,
            use_difference=self.use_difference,
            metric=self.metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            Xb=legacy_Xb,
            yb=legacy_yb,
        )

        if X_pairs.ndim != 2:
            raise ValueError('Generated pairwise features must be a 2D array.')
        if y_pairs.ndim != 1:
            y_pairs = y_pairs.reshape(-1)
        if X_pairs.shape[0] != y_pairs.shape[0]:
            raise ValueError('Pairwise features and targets must have the same number of rows.')
        if X_pairs.shape[0] < 1:
            raise ValueError('No pairwise samples were generated.')

        n_val = int(self.val_size * X_pairs.shape[0])
        perm = np.random.permutation(X_pairs.shape[0])
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_train, X_val = np.ascontiguousarray(X_pairs[train_idx]), np.ascontiguousarray(X_pairs[val_idx])
        y_train, y_val = y_pairs[train_idx], y_pairs[val_idx]

        if self.base_pred == 'zero':
            self.base_prediction_ = .0
        else:
            self.base_prediction_ = np.mean(y_pairs)
        residuals = y_train - self.base_prediction_
        residuals_val = y_val - self.base_prediction_

        prediction = np.ones((X_train.shape[0],)) * self.base_prediction_
        prediction_val = np.ones((X_val.shape[0],)) * self.base_prediction_

        self.estimators_ = []
        wait = 0
        for _ in tqdm(range(self.max_n_iter), position=0, leave=False, disable=not self.verbose):
            best_feat_idx = (-1,)
            best_est = None
            best_score, best_score_val = np.inf, np.inf

            # avoid empty minibatches when the train size is small
            n_inner_rows = max(1, int(X_train.shape[0] * .1))
            row_idx = np.random.choice(X_train.shape[0], n_inner_rows, replace=False)

            if self.n_jobs == 1:
                for feat_idx in tqdm(self._get_combinations(X_train, residuals), position=1, leave=False,
                                     disable=not self.verbose):
                    # pass the symmetry arguments also in the single thread path
                    score, score_val, feat_idx, est = _rulecardGAM_innerloop(
                        self.base_estimator, row_idx, feat_idx, self.learning_rate, X_train, X_val,
                        residuals, residuals_val,
                        self.enforce_simmetry, self.original_n_features, self.score_metric)

                    if best_score > score:
                        best_score, best_score_val = score, score_val
                        best_est = est
                        best_feat_idx = feat_idx

            else:
                combinations = list(self._get_combinations(X_train, residuals))
                results = Parallel(n_jobs=self.n_jobs, prefer="processes", verbose=0 if not self.verbose else 5)(
                    delayed(_rulecardGAM_innerloop)(self.base_estimator, row_idx, feat_idx,
                                                    self.learning_rate, X_train, X_val, residuals,
                                                    residuals_val, self.enforce_simmetry,
                                                    self.original_n_features, self.score_metric)
                    for feat_idx in combinations
                )

                for score, score_val, feat_idx, est in results:
                    if best_score > score:
                        best_score, best_score_val = score, score_val
                        best_feat_idx = feat_idx
                        best_est = est

            if best_feat_idx == (-1,):
                return self
            best_res_delta = _predict_residual_delta(best_est, X_train, best_feat_idx, self.learning_rate)
            best_res_delta_val = _predict_residual_delta(best_est, X_val, best_feat_idx, self.learning_rate)
            best_prediction = prediction + best_res_delta
            best_prediction_val = prediction_val + best_res_delta_val
            prev_score_val = _compute_score(y_val, prediction_val, self.score_metric)
            best_score_val = _compute_score(y_val, best_prediction_val, self.score_metric)

            if self.verbose:
                print(best_score_val < prev_score_val, best_score_val <= 10 ** -3,
                      best_score_val,
                      prev_score_val,
                      sep='\t')

            if best_score_val < prev_score_val or best_score_val <= 10 ** -3:
                wait = 0
                prediction_val = best_prediction_val
                residuals_val = residuals_val - best_res_delta_val
            else:
                wait += 1
                if wait >= self.patience:
                    self.estimators_ = self.estimators_[:-wait]
                    return self

            prediction = best_prediction
            residuals = residuals - best_res_delta

            if not self.reuse_features:
                self.used_features_.add(best_feat_idx)
                self.used_features_.add(tuple([x + self.original_n_features for x in best_feat_idx]))
            self.estimators_.append((best_feat_idx, best_est))

        return self


def _cosine_distance(a, b, eps=1e-12):
    """norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a <= eps and norm_b <= eps:
        return .0
    if norm_a <= eps or norm_b <= eps:
        return 1.0"""

    return cosine(a, b)


def _compute_score(y_true, y_pred, score_metric):
    if score_metric == 'mae':
        return np.mean(np.abs(y_true - y_pred))
    if score_metric == 'cosine':
        return _cosine_distance(y_true, y_pred)
    raise ValueError(f"Unknown score_metric '{score_metric}'.")


def _rulecardGAM_innerloop(base_estimator, row_idx, feat_idx, learning_rate,
                           X_train, X_val,
                           residuals,
                           residuals_val,
                           simmetry,
                           original_n_features,
                           score_metric
                           ):
    est = copy.deepcopy(base_estimator)

    if simmetry and original_n_features * 2 < min(feat_idx):
        feat_idx_inv = tuple([(x + original_n_features // 2) % original_n_features for x in feat_idx])
        X_train_all = np.vstack([
            X_train[np.ix_(row_idx, feat_idx)],
            X_train[np.ix_(row_idx, feat_idx_inv)],
        ]).reshape(len(row_idx) * 2, -1)

        est.fit(X_train_all, np.hstack([residuals[row_idx], residuals[row_idx]]))
    else:
        est.fit(X_train[np.ix_(row_idx, feat_idx)], residuals[row_idx])

    res_delta = _predict_residual_delta(est, X_train, feat_idx, learning_rate)
    res_delta_val = _predict_residual_delta(est, X_val, feat_idx, learning_rate)

    score = _compute_score(residuals, res_delta, score_metric)
    score_val = _compute_score(residuals_val, res_delta_val, score_metric)

    return score, score_val, feat_idx, est


def _predict_residual_delta(estimator, X, feat_idx, learning_rate):
    return learning_rate * estimator.predict(X[:, feat_idx].reshape(X.shape[0], -1))


def _resolve_subsample_size(n_instances, subsample):
    if isinstance(subsample, float):
        if subsample <= 0:
            raise ValueError('subsample must be > 0.')
        n_sub = int(round(n_instances * subsample))
    elif isinstance(subsample, (int, np.integer)):
        if subsample <= 0:
            raise ValueError('subsample must be > 0.')
        n_sub = int(subsample)
    else:
        raise ValueError('subsample must be a float or int.')

    return min(max(1, n_sub), n_instances)


def _extract_precomputed_targets(pair_targets, idx_a, idx_b, n_a, n_b):
    pair_targets = np.asarray(pair_targets)

    if pair_targets.ndim == 2:
        if pair_targets.shape != (n_a, n_b):
            raise ValueError(
                f'pair_targets matrix has shape {pair_targets.shape}, expected ({n_a}, {n_b}).')
        return pair_targets[idx_a, idx_b].astype(float)

    if pair_targets.ndim == 1:
        if pair_targets.shape[0] == len(idx_a):
            return pair_targets.astype(float)

        if pair_targets.shape[0] == n_a * n_b:
            flat_idx = idx_a * n_b + idx_b
            return pair_targets[flat_idx].astype(float)

        raise ValueError(
            f'pair_targets vector has length {pair_targets.shape[0]}, expected {len(idx_a)} or {n_a * n_b}.')

    raise ValueError('pair_targets must be a 1D vector or a 2D matrix.')


def _apply_pair_feature_view(X_pairwise, n_features, use_pairwise=True, use_difference=False):
    if not use_pairwise and not use_difference:
        raise ValueError('At least one of use_pairwise or use_difference must be True.')

    if X_pairwise.shape[1] < 2 * n_features:
        raise ValueError('Pairwise feature matrix has inconsistent shape.')

    X_pairwise = X_pairwise[:, :2 * n_features]

    if not use_difference:
        return X_pairwise

    X_diff = np.abs(X_pairwise[:, :n_features] - X_pairwise[:, n_features:])
    if use_pairwise:
        return np.hstack([X_pairwise, X_diff])

    return X_diff


def _gen_pairs(X: np.ndarray, y=None,
               mode='distance', model=None, pair_targets=None,
               subsample_strategy='random', subsample=1.,
               use_pairwise=True, use_difference=False,
               metric='euclidean', random_state=42, n_jobs=1,
               Xb=None, yb=None):
    """Generate pairwise features and targets according to the selected mode."""
    mode = mode.lower()

    if mode in {'distance', 'self_supervised'} and y is not None and Xb is None and pair_targets is None:
        X_pairs = np.asarray(X)
        y_pairs = np.asarray(y).reshape(-1)

        if X_pairs.ndim != 2:
            raise ValueError('X must be 2D when passing precomputed pairwise features.')
        if y_pairs.shape[0] != X_pairs.shape[0]:
            raise ValueError('y must have same number of rows as X when passing precomputed pairwise data.')

        if use_pairwise:
            if X_pairs.shape[1] % 2 != 0:
                raise ValueError('When use_pairwise=True, precomputed X must have an even number of columns.')
            n_features = X_pairs.shape[1] // 2
            X_pairs = _apply_pair_feature_view(X_pairs, n_features, use_pairwise=use_pairwise,
                                               use_difference=use_difference)
        elif use_difference:
            if X_pairs.shape[1] % 2 == 0:
                n_features = X_pairs.shape[1] // 2
                X_pairs = _apply_pair_feature_view(X_pairs, n_features, use_pairwise=False, use_difference=True)
            else:
                X_pairs = np.asarray(X_pairs)

        return X_pairs, y_pairs

    Xa = np.asarray(X)
    Xb_data = Xa if Xb is None else np.asarray(Xb)

    if Xa.ndim != 2 or Xb_data.ndim != 2:
        raise ValueError('X and Xb must be 2D arrays.')
    if Xa.shape[1] != Xb_data.shape[1]:
        raise ValueError('Xa and Xb must have same number of features.')

    if subsample_strategy in ['random', None]:
        # Keep legacy RNG side effects for backward-compatible training dynamics.
        np.random.seed(random_state)
        n_instances_Xa = _resolve_subsample_size(Xa.shape[0], subsample)
        n_instances_Xb = _resolve_subsample_size(Xb_data.shape[0], subsample)

        subset_idx_Xa = np.random.choice(Xa.shape[0], n_instances_Xa, replace=False)
        subset_idx_Xb = np.random.choice(Xb_data.shape[0], n_instances_Xb, replace=False)

        Xa_sub, Xb_sub = Xa[subset_idx_Xa], Xb_data[subset_idx_Xb]
    elif subsample_strategy == 'smart':
        raise ValueError('Not implemented.')
    else:
        raise ValueError('Invalid subsample strategy.')

    n_features = Xa_sub.shape[1]
    X_pairwise = np.hstack([np.repeat(Xa_sub, Xb_sub.shape[0], axis=0), np.tile(Xb_sub, (Xa_sub.shape[0], 1))])

    idx_a = np.repeat(subset_idx_Xa, len(subset_idx_Xb))
    idx_b = np.tile(subset_idx_Xb, len(subset_idx_Xa))

    if mode == 'distance':
        if pair_targets is not None:
            y_pairs = _extract_precomputed_targets(pair_targets, idx_a, idx_b, Xa.shape[0], Xb_data.shape[0])
        elif y is not None and yb is not None:
            ya = np.asarray(y).reshape(-1)
            yb = np.asarray(yb).reshape(-1)
            if ya.shape[0] != Xa.shape[0] or yb.shape[0] != Xb_data.shape[0]:
                raise ValueError('Legacy ya/yb must match Xa/Xb rows.')

            _, y_pairs = generate_pairwise_dataset(
                Xa_sub, Xb_sub, metric,
                ya=ya[subset_idx_Xa], yb=yb[subset_idx_Xb],
                n_jobs=n_jobs
            )
        else:
            _, y_pairs = generate_pairwise_dataset(Xa_sub, Xb_sub, metric=metric, n_jobs=n_jobs)

    elif mode == 'self_supervised':
        if pair_targets is not None:
            y_pairs = _extract_precomputed_targets(pair_targets, idx_a, idx_b, Xa.shape[0], Xb_data.shape[0])
        else:
            _, y_pairs = generate_pairwise_dataset(Xa_sub, Xb_sub, metric=metric, n_jobs=n_jobs)

    elif mode == 'supervised':
        if y is None:
            raise ValueError("mode='supervised' requires y labels.")

        labels_a = np.asarray(y).reshape(-1)
        labels_b = labels_a if yb is None else np.asarray(yb).reshape(-1)

        if labels_a.shape[0] != Xa.shape[0] or labels_b.shape[0] != Xb_data.shape[0]:
            raise ValueError('Label arrays must match the number of rows in X and Xb.')

        # Default binary pair target; easy to replace with continuous variants later.
        y_pairs = (labels_a[idx_a] != labels_b[idx_b]).astype(float)

    elif mode == 'model':
        if model is None:
            raise ValueError("mode='model' requires a fitted model exposing predict(X).")

        pred_a = np.asarray(model.predict(Xa)).reshape(-1)
        pred_b = pred_a if Xb is None else np.asarray(model.predict(Xb_data)).reshape(-1)

        if pred_a.shape[0] != Xa.shape[0] or pred_b.shape[0] != Xb_data.shape[0]:
            raise ValueError('model.predict(X) returned arrays with incompatible shapes.')

        y_pairs = np.abs(pred_a[idx_a] - pred_b[idx_b]).astype(float)

    elif mode == 'precomputed':
        if pair_targets is None:
            raise ValueError("mode='precomputed' requires pair_targets.")
        y_pairs = _extract_precomputed_targets(pair_targets, idx_a, idx_b, Xa.shape[0], Xb_data.shape[0])

    else:
        raise ValueError(f"Invalid mode '{mode}'.")

    y_pairs = np.asarray(y_pairs).reshape(-1)
    if y_pairs.shape[0] != X_pairwise.shape[0]:
        raise ValueError('Generated pair targets have inconsistent length.')

    X_out = _apply_pair_feature_view(X_pairwise, n_features,
                                     use_pairwise=use_pairwise,
                                     use_difference=use_difference)

    return X_out, y_pairs


# -----------------------------------------------------------------------------
# Usage examples
# -----------------------------------------------------------------------------
# from sklearn.ensemble import RandomForestRegressor
#
# X: (n_samples, n_features)
# y_cls: classification labels
#
# 1) Distance mode (target = pairwise metric distance)
# prc = PairwiseRuleCardGAM(metric='euclidean', random_state=42)
# prc.fit(X, mode='distance')
#
# 2) Supervised mode (0 same class, 1 different class)
# prc = PairwiseRuleCardGAM(random_state=42)
# prc.fit(X, y=y_cls, mode='supervised')
#
# 3) Model mode (target = abs difference between model predictions)
# base_model = RandomForestRegressor(random_state=42).fit(X, y_reg)
# prc = PairwiseRuleCardGAM(random_state=42)
# prc.fit(X, mode='model', model=base_model)
#
# 4) Precomputed mode (custom pair targets)
# custom_pair_targets = np.random.rand(X.shape[0], X.shape[0])
# prc = PairwiseRuleCardGAM(random_state=42)
# prc.fit(X, mode='precomputed', pair_targets=custom_pair_targets)
#
# 5) Self-supervised mode (fallback to metric when no target is provided)
# prc = PairwiseRuleCardGAM(metric='cosine', random_state=42)
# prc.fit(X, mode='self_supervised')


# interpretable learned distance via additive boosting + scorecard distillation
# interpretability is moved from the model to the metric
# clf e reg: knn
# clu: kmedoids, agglomerative, dbscan
# outliers: lof (abod?)
# similarity search? ranking? (anchor, positive) -> y = 0 (anchor, negative) -> y = 1
# as post-hoc explainer
# kernel similarity?
# riportare la distanza dallo spazio latente a quello delle feature (target=distanza spazio latente)

# future work
# integrare il task nel fitting della gam
