import numpy as np
from copy import copy
import pdb


def expected_utility_at_param(param, rollout, model_parameters, n_rollout_per_it):
    expected_utility = 0.
    for it in range(n_rollout_per_it):
        if isinstance(model_parameters, list):
            utility = rollout(param, model_parameters[it])
        else:
            utility = rollout(param, model_parameters)
        expected_utility += utility / n_rollout_per_it
    return expected_utility


def random_hill_climb_policy_optimizer(rollout, model_parameters, n_it=20, n_rollout_per_it=10, num_param=2):

    best_param = np.ones(num_param)

    # Get total utility at initial iterate
    best_utility = expected_utility_at_param(best_param, rollout, model_parameters, n_rollout_per_it=n_rollout_per_it)

    # Random search hill-climbing
    for it in range(n_it):
        param = best_param + np.random.normal(size=num_param)
        utility = expected_utility_at_param(param, rollout, model_parameters, n_rollout_per_it=n_rollout_per_it)
        if utility > best_utility:
            best_utility = utility
            best_param = param

    return best_param


def genetic_policy_optimizer(rollout, model_parameters, n_rollout_per_it=10, num_param=3, n_survive=5,
                             n_per_gen=10, n_gen=2):

    params = np.random.lognormal(size=(n_per_gen, num_param))
    for gen in range(n_gen):
        scores = np.ones(n_per_gen)
        for ix, p in enumerate(params):
            scores[ix] = expected_utility_at_param(p, rollout, model_parameters, n_rollout_per_it=n_rollout_per_it)
        params_to_keep = params[np.argsort(scores)[-n_survive:]]
        if gen < n_gen - 1:
            offspring_param_means = np.log(params_to_keep) - 1 / 2
            new_param_means = np.ones((n_per_gen - n_survive, num_param)) * -1 / 2
            param_means = np.vstack((offspring_param_means, new_param_means))
            params = np.random.lognormal(mean=param_means)
    best_param = params_to_keep[-1]
    return best_param


def random_q_optimizer(q, L, budget, n_it=100):
    A = np.zeros(L)
    A[:budget] = 1
    q_best = float('inf')
    A_best = None
    for it in range(n_it):
        np.random.shuffle(A)
        q_it = q(A)
        if q_it < q_best:
            q_best = q_it
            A_best = copy(A)
    return A_best, q_best


