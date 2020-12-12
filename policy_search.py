import numpy as np
from scipy.special import expit
from environment import PoissonDisease
from functools import partial


def priority_scores(policy_parameter, model_parameter, X, spatial_weight_matrix):
    # ToDo: fix computation of mean_counts
    mean_counts_ = np.dot(X, model_parameter)
    mean_counts_backup_ = np.dot(spatial_weight_matrix, mean_counts_)
    priority_features = np.array([mean_counts_, mean_counts_backup_])
    priority_score = expit(np.dot(priority_features, policy_parameter))
    return priority_score


def action_from_priority_scores(priority_scores, budget):
    L = len(priority_scores)
    A = np.zeros(L)
    priority_locations = np.argsort(priority_scores)[-budget:]
    A[priority_locations] = 1
    return A


def priority_score_policy(policy_parameter, model_parameter, budget, X, spatial_weight_matrix):
    prioritiy_scores_ = priority_scores(policy_parameter, model_parameter, X, spatial_weight_matrix)
    A = action_from_priority_scores(priority_scores_, budget)
    return A


def env_from_model_parameter(model_parameter, L):
    alpha_nu = np.log(model_parameter[0])
    beta_nu = model_parameter[1:2]
    alpha_lambda = np.log(model_parameter[3])
    lambda_a = model_parameter[4]
    alpha_phi = np.log(model_parameter[5])
    phi_a = model_parameter[6]

    env = PoissonDisease(L, lambda_a = lambda_a, phi_a = phi_a, alpha_nu = alpha_nu, alpha_lambda = alpha_lambda,
                         alpha_phi = alpha_phi, beta_nu = beta_nu)
    return env


def rollout(policy_parameter, model_parameter, budget, L, time_horizon, discount_factor=0.96):
    total_utility = 0.
    env = env_from_model_parameter(model_parameter, L)
    env.reset()

    # Initial action
    A = np.zeros(env.L)
    for t in range(time_horizon):
        env.step(A)
        total_utility -= discount_factor**t * env.Y.mean()
        X = env.X
        A = priority_score_policy(policy_parameter, model_parameter, budget, X, env.spatial_weight_matrix)

    return total_utility


def policy_search(model_parameter, budget, L, time_horizon, discount_factor, policy_optimizer):
    rollout_partial = partial(rollout, model_parameter=model_parameter, budget=budget, L=L, time_horizon=time_horizon,
                              discount_factor=discount_factor)
    policy_parameter_estimate = policy_optimizer(rollout_partial)
    return policy_parameter_estimate

