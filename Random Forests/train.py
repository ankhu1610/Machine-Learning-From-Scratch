from random import random
from sklearn  import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForests import RandomForest
data = datasets.load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
clf = RandomForest(n_trees= 20)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy
acc = accuracy(y_test, prediction)
print(acc)