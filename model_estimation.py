import pdb
import numpy as np
from scipy.optimize import minimize
from functools import partial
from environment import PoissonDisease


def negative_log_likelihood(param_vec, Y_stacked, X_stacked):
    param_vec[3] = np.exp(param_vec[3])
    param_vec[5] = np.exp(param_vec[5])
    endemic_term = np.exp(np.dot(X_stacked[:, :2], param_vec[:2]))
    autoregressive_term = np.dot(X_stacked[:, 2:4], param_vec[2:4])
    spatiotemporal_term = np.dot(X_stacked[:, 4:], param_vec[4:])

    mean_counts_ = endemic_term + autoregressive_term + spatiotemporal_term
    mean_counts_ = np.maximum(mean_counts_, 0.01)
    log_counts = np.log(mean_counts_)
    log_lik = np.dot(Y_stacked, log_counts) - np.sum(mean_counts_)
    return -log_lik


def fit_model(env):
    initial_param = np.concatenate(([env.alpha_nu], env.beta_nu, [env.alpha_lambda], [np.log(env.lambda_a)],
                                    [env.alpha_phi], [np.log(env.phi_a)]))
    Y_stacked = np.hstack(env.Y_list)
    X_stacked = np.vstack(env.X_list)
    nll_partial = partial(negative_log_likelihood, Y_stacked=Y_stacked, X_stacked=X_stacked)
    result = minimize(nll_partial, x0=initial_param)
    pdb.set_trace()
    return result.x


if __name__ == "__main__":
    L = 50
    T = 20
    env = PoissonDisease(L)
    env.reset()
    for _ in range(T):
        env.step(np.random.binomial(1, 0.1, L))
    param_hat = fit_model(env)

