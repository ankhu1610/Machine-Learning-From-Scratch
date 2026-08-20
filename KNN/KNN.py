import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2) ** 2))
class KNeighborsClassifier:
    def __init__(self,k= 3):
        self.k = k

    def fit(self,X, Y):
        self.X_train = X
        self.Y_train = Y
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    def _predict(self, x):
        distance = [euclidean_distance(x, x1) for x1 in self.X_train]
        # get the closest k 
        k_indices = np.argsort(distance)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        val = Counter(k_nearest_labels).most_common()#Return a list of the n most common elements and their counts from the most common to the least. If n is omitted or None, most_common() returns all elements in the counter. Elements with equal counts are ordered in the order first encountered:
        return val[0][0]