import numpy as np
import docplex.mp
import docplex.mp.model as dcpm
from sklearn.linear_model import LinearRegression


def lp_max(coef, budget):
    """
    Solve min_a coef.A + intercept, subject to sum A = budget.
    :return:
    """
    L = len(coef)
    model = dcpm.Model(name="lp_max")
    vars = {i: model.binary_var(name="trt_{}".format(i)) for i in range(L)}
    obj = model.sum(vars[i]*float(coef[i]) for i in range(L))
    model.add_constraint(model.sum(vars[i] for i in range(L)) == budget)
    model.minimize(obj)
    sol = model.solve(url=None, key=None)
    argmax = np.array([int(sol.get_value(f'trt_{i}')) for i in range(L)])
    return argmax


def fit_linear_approximation_at_location(q_l, A_samples):
    """
    Fit linear approximation to q_l at actions in A_samples.

    :param q_l: replicates of local q-function evaluated at actions in A_samples
    :param A_samples: list of actions
    :return:
    """
    lm = LinearRegression()
    lm.fit(A_samples, q_l)
    coef = lm.coef_
    return coef


def fit_lp(q, L, budget, samples=100):
    """
    Fit linear approximation to q_l as a function of actions, for each l.

    :param q: Q-function to be approximately optimized by binary linear programming.
    :param L:
    :param budget:
    :param samples: Number of actions at which to evaluate Q-function.
    :return:
    """
    # Evaluate q at many actions
    A_dummy = np.zeros(L)
    A_dummy[:budget] = 1
    A_samples = np.array([np.random.permutation(A_dummy) for _ in range(samples)])
    q_samples = np.array([q(A) for A in A_samples])

    # Evaluate
    coef = np.zeros(L)
    intercept = 0.
    for l in range(L):
        q_l = q_samples[:, l]
        coef_l = fit_linear_approximation_at_location(q_l, A_samples)
        coef += coef_l

    return coef, intercept