import time


class GradientDescentDualityCriteria:
    def __init__(self, StepSizeChoice, stop_criteria, return_history=True, name=None):
        self.stop_criteria_f = stop_criteria
        self.name = name
        self.StepSizeChoice = StepSizeChoice
        self.return_history = return_history
        self.history = []

    def __call__(self, x0, f, gradf, N):
        pass

    def stop_criteria(self):
        pass

    def solve(self, x0, f, gradf, tol=1e-3, max_iter=1000):
        self.history = [(f.f(x0), time.time())]
        x = x0.copy()
        k = 0
        x_prev = None
        while x_prev is None or abs(self.stop_criteria_f(x) - f.f(x)) > tol:
            h = -gradf(x)
            alpha = self.StepSizeChoice(x, h, k, gradf, f)
            x_prev, x = x, x + alpha * h
            if self.return_history:
                self.history.append((f.f(x), time.time()))
            if k >= max_iter:
                break
            k += 1
        return x