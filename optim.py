import numpy as np
from copy import copy
import pdb


def expected_utility_at_param(param, rollout, n_rollout_per_it):
    expected_utility = 0.
    for _ in range(n_rollout_per_it):
        utility = rollout(param)
        expected_utility += utility / n_rollout_per_it
    return expected_utility


def random_hill_climb_policy_optimizer(rollout, n_it=20, n_rollout_per_it=10, num_param=2):

    best_param = np.ones(num_param)

    # Get initial total utility
    best_utility = expected_utility_at_param(best_param, rollout, n_rollout_per_it=n_rollout_per_it)

    # Random search hill-climbing
    for it in range(n_it):
        param = best_param + np.random.normal(size=num_param)
        utility = expected_utility_at_param(param, rollout, n_rollout_per_it=n_rollout_per_it)
        if utility > best_utility:
            best_utility = utility
            best_param = param
            print(f'it: {it} best utility: {best_utility} best param: {best_param}')

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


