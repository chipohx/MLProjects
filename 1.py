import pandas as pd
import numpy as np


class MyLineReg:
    def __init__(self, n_iter, learning_rate, weights=None, metric=None, reg=None, l1_coef=0, l2_coef=0):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_metric = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __str__(self):
        return 'MyLineReg class: n_iter=%d, learning_rate=%s' % (self.n_iter, self.learning_rate)

    def fit(self, X, y, verbose=0):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.ones(X.shape[1])

        for i in range(self.n_iter):
            y_pred = X.dot(self.weights)
            error = y - y_pred
            grad = (2 / len(y)) * X.T.dot(error)
            if np.sum(self.weights) == 0:
                sgn = 0
            elif np.sum(self.weights) > 0:
                sgn = 1
            else:
                sgn = -1

            if self.reg:
                if self.reg == 'l1':
                    reg1 = self.l1_coef * np.sum(np.abs(self.weights))
                    grad_reg1 = grad + self.l1_coef * sgn
                    mae = (1 / len(y)) * np.sum(np.abs(error)) + reg1
                    mse = (1 / len(y)) * np.sum(error ** 2) + reg1
                    rmse = np.sqrt(mse) + reg1
                    mape = (100 / len(y)) * np.sum(np.abs(error) / np.abs(y)) + reg1
                    r2 = 1 - (np.sum(error ** 2) / np.sum((y - np.mean(y)) ** 2)) + reg1
                    self.weights -= self.learning_rate * grad_reg1

                elif self.reg == 'l2':
                    reg2 = self.l2_coef * np.sum(self.weights ** 2)
                    grad_reg2 = grad + self.l2_coef * 2 * self.weights
                    mae = (1 / len(y)) * np.sum(np.abs(error)) + reg2
                    mse = (1 / len(y)) * np.sum(error ** 2) + reg2
                    rmse = np.sqrt(mse) + reg2
                    mape = (100 / len(y)) * np.sum(np.abs(error) / np.abs(y)) + reg2
                    r2 = 1 - (np.sum(error ** 2) / np.sum((y - np.mean(y)) ** 2)) + reg2
                    self.weights -= self.learning_rate * grad_reg2

                elif self.reg == 'elsticnet':
                    reg1 = self.l1_coef * np.sum(np.abs(self.weights))
                    reg2 = self.l2_coef * np.sum(self.weights ** 2)
                    elNet = reg1 + reg2
                    grad_elNet = grad + self.l1_coef * sgn + self.l2_coef * 2 * self.weights
                    mae = (1 / len(y)) * np.sum(np.abs(error)) + elNet
                    mse = (1 / len(y)) * np.sum(error ** 2) + elNet
                    rmse = np.sqrt(mse) + elNet
                    mape = (100 / len(y)) * np.sum(np.abs(error) / np.abs(y)) + elNet
                    r2 = 1 - (np.sum(error ** 2) / np.sum((y - np.mean(y)) ** 2)) + elNet
                    self.weights -= self.learning_rate * grad_elNet
            else:
                mae = (1 / len(y)) * np.sum(np.abs(error))
                mse = (1 / len(y)) * np.sum(error ** 2)
                rmse = np.sqrt(mse)
                mape = (100 / len(y)) * np.sum(np.abs(error) / np.abs(y))
                r2 = 1 - (np.sum(error ** 2) / np.sum((y - np.mean(y)) ** 2))
                self.weights -= self.learning_rate * grad

                # if verbose:
                #     if self.metric == 'mae':
                #         current_metric = mae
                #     elif self.metric == 'mse':
                #         current_metric = mse
                #     elif self.metric == 'rmse':
                #         current_metric = rmse
                #     elif self.metric == 'mape':
                #         current_metric = mape
                #     elif self.metric == 'r2':
                #         current_metric = r2
                #     print(f"Iteration {i} | {self.metric.upper()}: {current_metric}")

        # final metrics
        y_pred = X.dot(self.weights)
        error = y_pred - y
        mae = (1 / len(y)) * np.sum(np.abs(error))
        mse = (1 / len(y)) * np.sum(error ** 2)
        rmse = np.sqrt(mse)
        mape = (100 / len(y)) * np.sum(np.abs(error) / np.abs(y))
        r2 = 1 - (np.sum(error ** 2) / np.sum((y - np.mean(y)) ** 2))

        if self.metric == 'mae':
            self.best_metric = mae
        elif self.metric == 'mse':
            self.best_metric = mse
        elif self.metric == 'rmse':
            self.best_metric = rmse
        elif self.metric == 'mape':
            self.best_metric = mape
        elif self.metric == 'r2':
            self.best_metric = r2

        if self.reg == 'l1':
            reg1 = self.l1_coef * np.sum(np.abs(self.weights))
            grad_reg1 = grad + self.l1_coef * sgn
        elif self.reg == 'l2':
            reg2 = self.l2_coef * np.sum(self.weights ** 2)
            grad_reg2 = grad + self.l2_coef * 2 * self.weights
        elif self.reg == 'elsticnet':
            reg1 = self.l1_coef * np.sum(np.abs(self.weights))
            reg2 = self.l2_coef * np.sum(self.weights ** 2)
            elNet = reg1 + reg2
            grad_elNet = grad + self.l1_coef * sgn + self.l2_coef * 2 * self.weights
        else:
            grad = (2 / len(y)) * X.T.dot(error)

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

model = MyLineReg(n_iter=50, learning_rate=0.1, metric='mse')
model.fit(X, y, verbose=10)
print(model.get_best_score())
