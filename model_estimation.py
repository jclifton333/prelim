import pdb
import numpy as np
from scipy.optimize import minimize
from functools import partial
from environment import PoissonDisease
import copy


def random_minimizer(f, center_param, n_draws=100, scale=0.5):
    best_param = center_param
    best_value = f(center_param)
    p = len(center_param)
    for draw in range(n_draws):
        param = best_param + np.random.normal(scale=scale, size=p)
        value = f(param)
        if value < best_value:
            best_value = value
            best_param = param
    return best_param


def negative_log_likelihood(param_vec, Y_stacked, X_stacked, weights=None, penalty=1.):
    transformed_param = copy.copy(param_vec)
    transformed_param[2:] = np.exp(transformed_param[2:])
    l2 = np.mean(transformed_param ** 2)
    endemic_term = np.exp(np.dot(X_stacked[:, :2], transformed_param[:2]))
    autoregressive_term = np.dot(X_stacked[:, 3:5], transformed_param[3:5])
    spatiotemporal_term = np.dot(X_stacked[:, 5:7], transformed_param[5:7])
    confounder_term = X_stacked[:, 7] * transformed_param[7]
    mean_counts_ = endemic_term + autoregressive_term + spatiotemporal_term + confounder_term
    mean_counts_ = np.maximum(mean_counts_, 0.1)
    log_counts = np.log(mean_counts_)
    if weights is not None:
        sum_mean_counts_ = np.dot(weights, mean_counts_)
        log_counts = np.multiply(log_counts, weights)
    else:
        sum_mean_counts_ = np.sum(mean_counts_)
    log_lik = np.dot(Y_stacked, log_counts) - sum_mean_counts_
    nll = -log_lik + penalty * l2
    # nll = np.mean(np.abs(mean_counts_ - Y_stacked)) + penalty * l2
    return nll


def fit_model(env, perturb=True):
    initial_param = np.concatenate(([env.alpha_nu], np.log(env.beta_nu), [env.alpha_lambda], [np.log(env.lambda_a)],
                                    [env.alpha_phi], [np.log(env.phi_a)], [env.alpha_confounder]))
    Y_stacked = np.hstack(env.Y_list)
    X_stacked = np.vstack(env.X_list)
    if perturb:
        n = X_stacked.shape[0]
        weights = np.random.exponential(size=n)
    else:
        weights = None
    nll_partial = partial(negative_log_likelihood, Y_stacked=Y_stacked, weights=weights, X_stacked=X_stacked)
    result = minimize(nll_partial, x0=initial_param, method='L-BFGS-B')
    estimate = result.x
    # estimate = random_minimizer(nll_partial, initial_param)
    return estimate


if __name__ == "__main__":
    L = 50
    T = 1000
    env = PoissonDisease(L)
    env.reset()
    for _ in range(T):
        env.step(np.random.binomial(1, 1, L))
        print(env.Y.sum())
    param_hat = fit_model(env)

