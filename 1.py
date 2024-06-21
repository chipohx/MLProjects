import random

import numpy as np


class MyLineReg:
    def __init__(self, n_iter, learning_rate, weights=None, metric=None, reg=None, l1_coef=1, l2_coef=1, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_metric = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return 'MyLineReg class: n_iter=%d, learning_rate=%s' % (self.n_iter, self.learning_rate)

    def fit(self, X, y, verbose=0):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.ones(X.shape[1])
        random.seed(self.random_state)

        for i in range(1, self.n_iter + 1):
            if self.sgd_sample:
                if isinstance(self.sgd_sample, int):
                    sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
                    X_sgd = X[sample_rows_idx]
                if isinstance(self.sgd_sample, float):
                    part_size = int(len(X) * self.sgd_sample)
                    sample_rows_idx = random.sample(range(X.shape[0]), part_size)
                    X_sgd = X[sample_rows_idx]
            else:
                X_sgd = X

            y_pred = X.dot(self.weights)
            error = y_pred - y
            grad = (2 / len(y)) * X_sgd.T.dot(error)

            if self.reg:
                if self.reg == 'l1':
                    grad += self.l1_coef * np.sign(self.weights)
                elif self.reg == 'l2':
                    grad += self.l2_coef * 2 * self.weights
                elif self.reg == 'elasticnet':
                    grad += self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights

            if callable(self.learning_rate):
                lr = self.learning_rate(i)
            else:
                lr = self.learning_rate
            self.weights -= lr * grad

            if verbose:
                current_metric = self._calculate_metric(error)
                print(f"Iteration {i} | {self.metric.upper()}: {current_metric}")

        self.best_metric = self._calculate_metric(error)

    def _calculate_metric(self, error):
        mae = (1 / len(error)) * np.sum(np.abs(error))
        mse = (1 / len(error)) * np.sum(error ** 2)
        rmse = np.sqrt(mse)
        mape = (100 / len(error)) * np.sum(np.abs(error) / np.abs(error + np.finfo(float).eps))
        r2 = 1 - (np.sum(error ** 2) / np.sum((error - np.mean(error)) ** 2))

        if self.metric == 'mae':
            return mae
        elif self.metric == 'mse':
            return mse
        elif self.metric == 'rmse':
            return rmse
        elif self.metric == 'mape':
            return mape
        elif self.metric == 'r2':
            return r2

    def get_best_score(self):
        return self.best_metric

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        y_pred = X.dot(self.weights)
        return y_pred

