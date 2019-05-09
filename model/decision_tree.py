import numpy as np
import sys
sys.setrecursionlimit(10000)

class DecisionTree(object):
    def __init__(self, tree = 'cls', criterion = 'gini', prune = 'depth', max_depth = 4, min_criterion = 0.05):
        self._feature = None
        self._label = None
        self._n_samples = None
        self._gain = None
        self._left = None
        self._right = None
        self._threshold = 0
        self._depth = 0

        self._root = None
        self._criterion = criterion
        self._prune = prune
        self._max_depth = max_depth
        self._min_criterion = min_criterion
        self._tree = tree

    def fit(self, features, target):
        self._root = DecisionTree()
        if(self._tree == 'cls'):
            self._root._grow_tree(features, target, self._criterion)
        else:
            self._root._grow_tree(features, target, 'mse')
        self._root._prune(self._prune, self._max_depth, self._min_criterion, self._root._n_samples)

    def predict(self, features):
        return np.array([self._root._predict(f) for f in features])

    def print_tree(self):
        self._root._show_tree(0, ' ')

    def _grow_tree(self, features, target, criterion = 'gini'):
        self._n_samples = features.shape[0] 

        if len(np.unique(target)) == 1:
            self._label = target[0]
            return

        best_gain = 0.0
        best_feature = 0
        best_threshold = 0

        if criterion in {'gini', 'entropy'}:
            self._label = max([(c, len(target[target == c])) for c in np.unique(target)], key = lambda x : x[1])[0]
        else:
            self._label = np.mean(target)

        impurity_node = self._calc_impurity(criterion, target)
        
        for col in range(features.shape[1]):
            feature_level = np.unique(features[:,col])
            thresholds = (feature_level[:-1] + feature_level[1:]) / 2.0

            for threshold in thresholds:
                target_l = target[features[:,col] <= threshold]
                impurity_l = self._calc_impurity(criterion, target_l)
                n_l = float(target_l.shape[0]) / self._n_samples

                target_r = target[features[:,col] > threshold]
                impurity_r = self._calc_impurity(criterion, target_r)
                n_r = float(target_r.shape[0]) / self._n_samples

                impurity_gain = impurity_node - (n_l * impurity_l + n_r * impurity_r)
                if impurity_gain > best_gain:
                    best_gain = impurity_gain
                    best_feature = col
                    best_threshold = threshold

        self._feature = best_feature
        self._gain = best_gain
        self._threshold = best_threshold
        self._split_tree(features, target, criterion)

    def _split_tree(self, features, target, criterion):
        features_l = features[features[:, self._feature] <= self._threshold]
        target_l = target[features[:, self._feature] <= self._threshold]
        self._left = DecisionTree()
        self._left.depth = self._depth + 1
        self._left._grow_tree(features_l, target_l, criterion)

        features_r = features[features[:, self._feature] > self._threshold]
        target_r = target[features[:, self._feature] > self._threshold]
        self._right = DecisionTree()
        self._right.depth = self._depth + 1
        self._right._grow_tree(features_r, target_r, criterion)

    def _calc_impurity(self, criterion, target):
        if criterion == 'gini':
            return 1.0 - sum([(float(len(target[target == c])) / float(target.shape[0])) ** 2.0 for c in np.unique(target)])
        elif criterion == 'mse':
            return np.mean((target - np.mean(target)) ** 2.0)
        else:
            entropy = 0.0
            for c in np.unique(target):
                p = float(len(target[target == c])) / target.shape[0]
                if p > 0.0:
                    entropy -= p * np.log2(p)
            return entropy            

    def _prune(self, method, max_depth, min_criterion, n_samples):
        if self.feature is None:
            return

        self._left._prune(method, max_depth, min_criterion, n_samples)
        self._right._prune(method, max_depth, min_criterion, n_samples)

        pruning = False

        if method == 'impurity' and self._left.feature is None and self._right.feature is None: 
            if (self.gain * float(self._n_samples) / n_samples) < min_criterion:
                pruning = True
        elif method == 'depth' and self._depth >= max_depth:
            pruning = True

        if pruning is True:
            self._left = None
            self._right = None
            self._feature = None

    def _predict(self, d):
        if self._feature != None:
            if d[self._feature] <= self._threshold:
                return self._left._predict(d)
            else:
                return self._right._predict(d)
        else: 
            return self._label

    def _show_tree(self, depth, cond):
        base = '    ' * depth + cond
        if self._feature != None:
            print(base + 'if X[' + str(self._feature) + '] <= ' + str(self._threshold))
            self._left._show_tree(depth+1, 'then ')
            self._right._show_tree(depth+1, 'else ')
        else:
            print(base + '{value: ' + str(self._label) + ', samples: ' + str(self._n_samples) + '}')