import numpy as np
from scipy.special import expit
from sklearn.base import ClassifierMixin, BaseEstimator, RegressorMixin

from PairwiseRuleCard.PairwiseRuleCardGAM import PairwiseRuleCardGAM


class PairwiseScoreCard(RegressorMixin, BaseEstimator):
    def __init__(self, rules, lr, base_log_odds, classes, PDO=20, odds0=50, score0=600, aggregate=True):
        self.rules = rules
        self.lr = lr
        self.base_log_odds = base_log_odds
        self.classes = classes
        self.PDO = PDO
        self.odds0 = odds0
        self.score0 = score0

        self.factor = PDO / np.log(2)
        self.offset = score0 - self.factor * np.log(odds0)
        self.base_points = round(self.offset + self.factor * self.base_log_odds)
        self.scorecard = []

        for rule, pred_residuals, support, log_odds in rules:
            w = log_odds * lr
            points = round(self.factor * w)
            if points == 0:
                continue
            self.scorecard.append([self.base_points, rule, support, points])

        if aggregate:
            tmp = dict()
            for base_points, rule, support, points in self.scorecard:
                k = tuple(rule)
                if k in tmp:
                    tmp[k] = (base_points, rule, max(tmp[k][2], support), points + tmp[k][-1])
                else:
                    tmp[k] = (base_points, rule, support, points)

            self.scorecard = list(tmp.values())


    @classmethod
    def _get_rules(cls, node, real_idx, gamma_map, X):
        if node.is_leaf():
            return [([], node.prediction, len(X), gamma_map[node.node_id])]

        rules = []

        feature = real_idx[node.stump.feature_original[0]]
        threshold = node.stump.threshold_original[0]
        cat = node.stump.is_categorical

        if cat:
            X_l = X[X[:, feature] == threshold]
            X_r = X[X[:, feature] != threshold]
            op_l, op_r = "==", "!="
        else:
            X_l = X[X[:, feature] <= threshold]
            X_r = X[X[:, feature] > threshold]
            op_l, op_r = "<=", ">"

        if node.node_l is not None:
            for r, pred, support, log_odds in cls._get_rules(node.node_l, real_idx, gamma_map, X_l):
                rules.append(([(feature, threshold, op_l)] + r, pred, support, log_odds))

        if node.node_r is not None:
            for r, pred, support, log_odds in cls._get_rules(node.node_r, real_idx, gamma_map, X_r):
                rules.append(([(feature, threshold, op_r)] + r, pred, support, log_odds))

        return rules

    @classmethod
    def _get_all_rules(cls, clf: PairwiseRuleCardGAM, X):
        rules = []
        for est in clf.estimators_:
            if est[1].root.is_leaf():
                continue
            rules += cls._get_rules(est[1].root, est[0], est[2], X)

        return rules

    @classmethod
    def from_rulecard(cls, clf:PairwiseRuleCardGAM, X, PDO=20, odds0=50, score0=600):
        rules = cls._get_all_rules(clf, X)
        return cls(rules, clf.learning_rate, clf.base_log_odds, clf.classes_, PDO, odds0, score0)

    def _get_activation(self, X:np.ndarray, rules):
        activation = np.ones((X.shape[0],)).astype(np.bool)
        for feat_idx, thr, comp in rules:
            activation &= X[:, feat_idx] <= thr if comp == '<=' else X[:, feat_idx] > thr
        return activation

    def predict_proba(self, X:np.ndarray):
        scores = np.zeros((X.shape[0],))
        for base_points, rules, support, points in self.scorecard:
            scores[self._get_activation(X, rules)] += points
        scores += self.scorecard[0][0]  # basepoints
        return expit((scores - self.offset) / self.factor)

    def predict(self, X):
        return np.vectorize(lambda x: self.classes[x])((self.predict_proba(X) > .5).astype(int))
