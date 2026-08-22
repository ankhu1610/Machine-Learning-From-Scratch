import numpy as np 
from collections import Counter
class Node:
    def __init__(self,feature = None, threshold = None, left = None, right = None, *, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right 
        self.value = value
    def is_leaf_node(self):
        return True if self.value is not None else False 

class DecisionTree:
    def __init__(self, min_samples_split = 2, max_depth = 100, n_features = 2):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features

    def fit(self, X,y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X,y)


    def _grow_tree(self, X,y, depth = 0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)
        feat_idx = np.random.choice(n_feats, self.n_features, replace = False)
        best_feat, best_thresh = self._best_split(X,y,feat_idx)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)
    


    def _best_split(self, X,y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                gain = self._information_gain(y,X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        #parent loss
        parent_entropy = self._entropy(y)
        #children loss
        left_idx, right_idx = self._split(X_column, threshold)
        if(len(left_idx ) == 0 or len(right_idx) == 0):
            return 0;
        child_entropy = len(left_idx)/len(y)* self._entropy(y[left_idx]) + len(right_idx)/len(y)* self._entropy(y[right_idx]);
        return parent_entropy - child_entropy;


    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist/len(y)
        return -np.sum([p*np.log2(p) for p in ps if p>0])
    def _split(self, X_column, threshold):
            left_idx = np.argwhere(X_column < threshold).flatten()
            right_idx = np.argwhere(X_column >= threshold).flatten() 
            return left_idx, right_idx
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return np.array([self._prediction(self.root,x) for x in X])
        

    def _prediction(self, root, X):
        if root.is_leaf_node():
            return root.value
        elif X[root.feature] <root.threshold :
            return self._prediction(root.left, X)
        else:
            return self._prediction(root.right,X)
