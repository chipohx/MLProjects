import pandas as pd
import numpy as np


class MyLineReg:
    def __init__(self, n_iter, learning_rate, weights=None, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_metric = None

    def __str__(self):
        return 'MyLineReg class: n_iter=%d, learning_rate=%s' % (self.n_iter, self.learning_rate)

    def fit(self, X, y, verbose=0):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.ones(X.shape[1])

        for i in range(self.n_iter):
            y_pred = X.dot(self.weights)
            error = y - y_pred
            grad = (2 / len(y)) * X.T.dot(error)
            self.weights -= self.learning_rate * grad

            # Calculate metrics
            mae = (1 / len(y)) * np.sum(np.abs(error))
            mse = (1 / len(y)) * np.sum(error ** 2)
            rmse = np.sqrt(mse)
            mape = (100 / len(y)) * np.sum(np.abs(error) / np.abs(y))
            r2 = 1 - (np.sum(error ** 2) / np.sum((y - np.mean(y)) ** 2))

            if self.metric == 'mae':
                self.best_metric = mae
                # print(f"Iteration {i} | Loss: {MSE} | MAE: {mae}")
            elif self.metric == 'mse':
                self.best_metric = mse
                # print(f"Iteration {i} | Loss: {MSE} | MSE: {mse}")
            elif self.metric == 'rmse':
                self.best_metric = rmse
                # print(f"Iteration {i} | Loss: {MSE} | RMSE: {rmse}")
            elif self.metric == 'mape':
                self.best_metric = mape
                # print(f"Iteration {i} | Loss: {MSE} | MAPE: {mape}")
            elif self.metric == 'r2':
                self.best_metric = r2
                # print(f"Iteration {i} | Loss: {MSE} | R2: {r2}")

    def get_best_score(self):
        return self.best_metric

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        y_pred = X.dot(self.weights)
        return y_pred


X = pd.DataFrame([[12, 0, 1], [0, 0, 2]])
y = pd.Series([12, 2])

model = MyLineReg(n_iter=50, learning_rate=0.1, metric='mape')
model.fit(X, y, verbose=10)
print(model.get_best_score())
