import math

import numpy as np


class LogisticRegressionFun:
    """
    Logistic Regression function

    ...

    Attributes
    ----------
    X : matrix
        Exposure in seconds.

    y :
    Methods
    -------
    f(omega)
        call function

    gradient(X, y)
        invoke gradient

    """
    def __init__(self, X, y, alpha=0.001):
        self.X = X
        self.y = y
        self.alpha = alpha

    def f(self, omega):
        exp_power = -self.y * self.X.dot(omega)
        max_exp_power = np.max(exp_power)
        # exp_power = (exp_power - np.min(exp_power)) / (np.max(exp_power) - np.min(exp_power))

        if max_exp_power < 30:
            first_term = np.mean(
                max_exp_power + np.log(1 / math.exp(max_exp_power) + np.exp(exp_power - max_exp_power)))
        else:
            first_term = np.mean(exp_power)

        second_term = (self.alpha / 2) * np.linalg.norm(omega, ord=2) ** 2
        return first_term + second_term

    def gradient(self, X, y):
        alpha = self.alpha
        m = X.shape[0]

        def f(omega):
            exp_power = -y * X.dot(omega)
            # Normalize to avoid overflow
            exp_power = (exp_power - np.min(exp_power)) / (np.max(exp_power) - np.min(exp_power))

            sigmoid_term = 1 / (1 + np.exp(exp_power))
            gradient_without_reg = -1 / m * X.T.dot(y * sigmoid_term)
            regularization_term = alpha * omega
            gradient = gradient_without_reg + regularization_term

            return gradient

        return lambda omega: f(omega)


class DualityGapLogisticRegression:
    """
    Duality function is min_x g(x, \nu, \lambda). We use it for duality gap.
    Duality gap is function minimum criteria: f(x) - g(x)

    ...

    Attributes
    ----------
    X : matrix
        Exposure in seconds.

    y :
    Methods
    -------
    f(omega)
        call function

    gradient(X, y)
        invoke gradient

    """
    def __init__(self, X, y, alpha=0.01):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.inverse_one_dot_x = 1 / np.sum(self.X, axis=1)
        self.inverse_X = np.linalg.pinv(X)

    def f(self, omega):
        nu = -1 * (self.alpha / self.y) * np.dot(omega, self.inverse_X)
        nu += 1e-6
        yX = self.y[:, np.newaxis] * self.X
        m = self.y.shape[0]

        return (m**-1 * np.sum(np.log(1 + nu * m / (1 - nu*m))) +
                (2*self.alpha)**-1 * np.linalg.norm(np.dot(nu, yX))**2 +
                np.dot(nu, np.log(nu * m / (1 - nu*m)) + np.dot(yX, self.alpha**-1 * np.dot(nu, yX))))

    def gradient(self, X, y):
        assert 0, "Not implemented yet"
