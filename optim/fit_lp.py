import numpy as np
from sklearn.linear_model import LinearRegression


def fit_linear_approximation_at_location(q_l, A_samples):
    lm = LinearRegression()
    lm.fit(A_samples, q_l)
    coef = lm.coef_
    intercept = lm.intercept
    return coef, intercept


def fit_lp(q, L, budget, samples=100):
    # Evaluate q at many actions
    A_dummy = np.zeros(L)
    A_dummy[:budget] = 1
    A_samples = np.array([np.random.permutation(A_dummy) for _ in range(samples)])
    q_samples = np.array([q(A) for A in A_samples])

    # Evaluate
    coef = np.zeros(env.L)
    intercept = 0.
    for l in range(L):
        q_l = q_samples[:, l]
        coef_l, intercept_l = fit_linear_approximation_at_location(q_l, A_samples)
        coef += coef_l
        intercept += intercept_l





