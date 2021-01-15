import numpy as np
from scipy.optimize import minimize
from functools import partial
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


def negative_log_likelihood(param_vec, Y_stacked, X_stacked, K_stacked, weights=None, penalty=1.):
    transformed_param = copy.copy(param_vec)
    transformed_param[3:] = np.exp(transformed_param[3:])
    l2 = np.mean(transformed_param ** 2)
    endemic_term = np.exp(np.dot(X_stacked[:, :3], transformed_param[:3]))
    autoregressive_term = np.dot(X_stacked[:, 3:5], transformed_param[3:5])
    spatiotemporal_term = np.dot(K_stacked[:, :2], transformed_param[5:7])
    mean_counts_ = endemic_term + autoregressive_term + spatiotemporal_term
    mean_counts_ = np.maximum(mean_counts_, 0.1)
    log_counts = np.log(mean_counts_)
    if weights is not None:
        sum_mean_counts_ = np.dot(weights, mean_counts_)
        log_counts = np.multiply(log_counts, weights)
    else:
        sum_mean_counts_ = np.sum(mean_counts_)
    log_lik = np.dot(Y_stacked, log_counts) - sum_mean_counts_
    nll = -log_lik + penalty * l2
    return nll


def fit_model(env, kernel='network', perturb=True):
    """
    Fit PoissonDisease model using penalized maximum likelihood estimation from observation history in env
    and assuming features are constructed with specified kernel.
    """

    initial_param = np.concatenate(([env.alpha_nu], np.log(env.beta_nu), [env.alpha_lambda], [np.log(env.lambda_a)],
                                    [env.alpha_phi], [np.log(env.phi_a)]))
    Y_stacked = np.hstack(env.Y_list)
    X_stacked = np.vstack(env.X_list)
    K_list = env.get_K_history(kernel)
    K_stacked = np.vstack(K_list)

    if perturb:
        n = X_stacked.shape[0]
        weights = np.random.exponential(size=n)
    else:
        weights = None

    nll_partial = partial(negative_log_likelihood, Y_stacked=Y_stacked, weights=weights, X_stacked=X_stacked,
                          K_stacked=K_stacked)
    result = minimize(nll_partial, x0=initial_param, method='L-BFGS-B')
    estimate = result.x
    estimate = np.maximum(np.minimum(estimate, 3), -3)  # clamp estimate to sane range
    return estimate
