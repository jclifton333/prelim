import numpy as np
import numba as nb


def negative_log_likelihood(param_vec, Y_stacked, X_stacked):
    mean_counts_ = np.dot(X_stacked, param_vec)
    log_counts = np.log(mean_counts_)
    log_lik = np.dot(Y_stacked, log_counts) - np.sum(mean_counts_)
    return -log_lik
