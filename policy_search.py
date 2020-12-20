import numpy as np
from scipy.special import expit
from environment import PoissonDisease
from model_estimation import fit_model
from functools import partial
from optim import random_hill_climb_policy_optimizer
from numba import njit
import pdb


def priority_scores(policy_parameter, model_parameter, X, spatial_weight_matrix):
    mean_counts_ = mean_counts_from_model_parameter(model_parameter, X)
    mean_counts_backup_ = np.dot(spatial_weight_matrix, mean_counts_)
    priority_features = np.column_stack([mean_counts_, mean_counts_backup_])
    priority_score = expit(np.dot(priority_features, policy_parameter))
    return priority_score


def mean_counts_from_model_parameter(model_parameter, X):
    # ToDo: handling of parameter transformations can probably be improved here and in env_from_param

    alpha_nu = model_parameter[0]
    beta_nu = np.exp(model_parameter[1:3])
    endemic_param = np.concatenate([[alpha_nu], beta_nu])
    endemic_term = np.dot(X[:, 0:3], endemic_param)
    endemic_term = np.exp(endemic_term)

    autoregressive_term = np.dot(X[:, 3:5], np.exp(model_parameter[3:5]))
    spatiotemporal_term = np.dot(X[:, 5:7], np.exp(model_parameter[5:7]))
    confounder_term = X[:, 7] * model_parameter[7]

    mean_counts_ = endemic_term + autoregressive_term + spatiotemporal_term + confounder_term
    return mean_counts_


def action_from_priority_scores(priority_scores_, budget):
    L = len(priority_scores_)
    A = np.zeros(L)
    priority_locations = np.argsort(priority_scores_)[-budget:]
    A[priority_locations] = 1
    return A


def priority_score_policy(policy_parameter, model_parameter, budget, X, spatial_weight_matrix):
    priority_scores_ = priority_scores(policy_parameter, model_parameter, X, spatial_weight_matrix)
    A = action_from_priority_scores(priority_scores_, budget)
    return A


def env_from_model_parameter(model_parameter, Y_current, t_current, L):
    alpha_nu = model_parameter[0]
    beta_nu = np.exp(model_parameter[1:3])
    alpha_lambda = model_parameter[3]
    lambda_a = np.exp(model_parameter[4])
    alpha_phi = model_parameter[5]
    phi_a = np.exp(model_parameter[6])
    alpha_confounder = model_parameter[7]

    env = PoissonDisease(L, lambda_a = lambda_a, phi_a = phi_a, alpha_nu = alpha_nu, alpha_lambda = alpha_lambda,
                         alpha_phi = alpha_phi, beta_nu = beta_nu, alpha_confounder=alpha_confounder,
                         Y_initial=Y_current, t_initial=t_current)
    return env


def rollout(policy_parameter, env, budget, time_horizon, discount_factor=0.96):
    total_utility = 0.
    model_parameter = fit_model(env, perturb=True)
    rollout_env = env_from_model_parameter(model_parameter, env.Y, env.t, env.L)
    rollout_env.reset()

    A = np.zeros(rollout_env.L) # Initial action
    for t in range(time_horizon):
        rollout_env.step(A)
        total_utility -= discount_factor**t * rollout_env.Y.mean()
        X = rollout_env.X
        A = priority_score_policy(policy_parameter, model_parameter, budget, X, rollout_env.spatial_weight_matrix)

    return total_utility


def policy_search(env, budget, time_horizon, discount_factor, policy_optimizer):
    rollout_partial = partial(rollout, env=env, budget=budget, time_horizon=time_horizon, discount_factor=discount_factor)
    policy_parameter_estimate = policy_optimizer(rollout_partial)
    return policy_parameter_estimate


def policy_search_policy(env, budget, time_horizon, discount_factor,
                         policy_optimizer=random_hill_climb_policy_optimizer):
    model_parameter_estimate = fit_model(env, perturb=False)
    policy_parameter_estimate = policy_search(env, budget, time_horizon, discount_factor, policy_optimizer)
    A = priority_score_policy(policy_parameter_estimate, model_parameter_estimate, budget, env.X,
                              env.spatial_weight_matrix)
    return A


if __name__ == "__main__":
    L = 50
    time_horizon = 10
    budget = 10
    discount_factor = 0.96
    policy_optimizer = random_hill_climb_policy_optimizer

    env = PoissonDisease(L=L)
    env.reset()

    A = np.zeros(L)
    env.step(A)
    total_reward = 0.
    for t in range(time_horizon):
        total_reward += discount_factor**t * env.Y.mean()
        A = policy_search_policy(env, budget, time_horizon-t, discount_factor, policy_optimizer)
        env.step(A)
        print(t, total_reward)

