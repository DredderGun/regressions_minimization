import unittest

import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        # L = 3
        # N = 1000
        # x0 = np.random.rand(X.shape[1])
        # grad = gradf(X, y)
        # res_chat = grad(np.ones(X.shape[1]))
        # method = GradientDescentDualityCriteria(ConstantStepSize(1 / L), name="GD, 1/L")
        # solve = method.solve(x0, f, gradf(X, y), tol=1e-9, max_iter=1000)

        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
