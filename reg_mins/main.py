import numpy as np
from scipy import sparse
import matplotlib
import matplotlib.pyplot as plt

from reg_mins.algorithms import GradientDescentDualityCriteria
from reg_mins.functions import DualityGapLogisticRegression, LogisticRegressionFun
from reg_mins.step_sizes import ConstantStepSize


def start_logistic_regression_duality():
    mn_X = np.load('./data/X.npy')
    mn_y = np.load('./data/y.npy')

    alpha = 0.001
    omega0 = np.random.rand(mn_X.shape[1])

    gap = DualityGapLogisticRegression(mn_X, mn_y, alpha=alpha)
    method = GradientDescentDualityCriteria(ConstantStepSize(0.1), gap.f)
    f = LogisticRegressionFun(mn_X, mn_y, alpha=alpha)
    method.solve(omega0, f, f.gradient(mn_X, mn_y), max_iter=1000)

    fig, ax = plt.subplots()
    x = []
    y = []
    for (val, i) in method.history:
        x.append(i)
        y.append(val)

    ax.plot(np.array(x), np.array(y))

    plt.savefig('plot.png', bbox_inches='tight')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_logistic_regression_duality()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
