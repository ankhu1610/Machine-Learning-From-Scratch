import numpy as np
def sigmoid(x):
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0# it will create something like [-2, -3,-1,0,1] as [false, false ,false, true, true]    
    negative = ~positive

    # Positive values
    out[positive] = 1/(1+ np.exp(-x[positive]))
    # Negative values
    out[negative] = np.exp(x[negative])/(1+np.exp(x[negative]))
    return out
class LogisticRegression:
    def __init__(self, lr = 0.001, n_iters = 100000):
        self.lr  = lr 
        self.n_iters = n_iters 
        self.weights = None
        self.bias = None
    def fit(self, X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights )+ self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions -y ))
            db = (1/n_samples) * np.sum(predictions -y)
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
    def predict(self,X):
        linear_pred = np.dot(X, self.weights)+ self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [1 if i> 0.5 else 0 for i in y_pred]
        return class_pred
    
   