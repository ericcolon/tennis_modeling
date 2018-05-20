import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class NonZeroLogit():

    def __init__(self, lmbda=0.1, prior=0., seed=None):
        self.lmbda = lmbda
        self.prior = prior
        self.seed = seed

    @staticmethod
    def loss(beta, X, y, lmbda, prior, w):
        # Negate likelihood because we lbfgs minimizes
        y_hat = sigmoid(X.dot(beta))
        return -np.sum(w *
            (y * np.log(y_hat) + (1. - y) * np.log(1. - y_hat))
        ) + lmbda * (beta - prior).transpose().dot(beta - prior)

    @staticmethod
    def fprime(beta, X, y, lmbda, prior, w):
        y_hat = sigmoid(X.dot(beta))
        return -(X.transpose().dot(w * (y - y_hat))) + lmbda * (beta - prior)

    def fit(self, X, y, sample_weight=None):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.beta = np.random.normal(size=X.shape[1])  # Initialize randomly
        if sample_weight is None:
            w = np.ones(X.shape[0])
        else:
            w = sample_weight
        result = fmin_l_bfgs_b(
            self.loss,
            x0 = self.beta,
            args = (X, y, self.lmbda, self.prior, w),
            fprime = self.fprime,
            pgtol =  1e-3,
            disp = True
        )
        self.beta = result[0]

    def predict_proba(self, X):
        p1 = sigmoid(X.dot(self.beta))
        return np.array([1. - p1, p1]).transpose()

    def predict(self, X):
        return (sigmoid(X.dot(self.beta)) > 0.5).astype(int)
