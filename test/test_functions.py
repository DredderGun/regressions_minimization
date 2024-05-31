import unittest

import numpy as np

from reg_mins.functions import LogisticRegressionFun, DualityGapLogisticRegression


class MyTestCase(unittest.TestCase):
    def test_f(self):
        np.random.seed(43)
        X = np.load('data/X.npy')
        y = np.load('data/y.npy')

        log_regr = LogisticRegressionFun(X, y)
        val = log_regr.f(np.random.rand(X.shape[1]))

        self.assertIsNotNone(val)

    def test_duality_gap(self):
        np.random.seed(43)
        X = np.load('data/X.npy')
        y = np.load('data/y.npy')

        omega = np.random.rand(X.shape[0]) * 10

        dual_f = DualityGapLogisticRegression(X, y)
        val = dual_f.f(omega)

        self.assertIsNotNone(val)


if __name__ == '__main__':
    unittest.main()
