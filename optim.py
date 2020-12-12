import numpy as np


def expected_utility_at_param(param, rollout, n_rollout_per_it):
    expected_utility = 0.
    for _ in range(n_rollout_per_it):
        utility = rollout(param)
        expected_utility += utility / n_rollout_per_it
    return expected_utility


def random_hill_climb_policy_optimizer(rollout, n_it=100, n_rollout_per_it=100, num_param=7):

    best_param = np.ones(num_param)

    # Get initial total utility
    best_utility = expected_utility_at_param(best_param, rollout, n_rollout_per_it=n_rollout_per_it)

    # Random search hill-climbing
    for _ in range(n_it):
        param = best_param + np.random.normal(size=num_param)
        utility = expected_utility_at_param(param, rollout, n_rollout_per_it=n_rollout_per_it)
        if utility > best_utility:
            best_utility = utility
            best_param = param

    return best_param

