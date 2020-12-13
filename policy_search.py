import numpy as np
from scipy.special import expit
from environment import PoissonDisease
from model_estimation import fit_model
from functools import partial
from optim import random_hill_climb_policy_optimizer


def priority_scores(policy_parameter, model_parameter, X, spatial_weight_matrix):
    alpha_nu = np.log(model_parameter[0])
    beta_nu = model_parameter[1:3]
    endemic_param = np.concatenate([[alpha_nu], beta_nu])
    endemic_term = np.dot(X[:3], endemic_param)
    endemic_term = np.exp(endemic_term)
    autoregressive_term = np.dot(X[3:5], model_parameter[3:5])
    spatiotemporal_term = np.dot(X[5:], model_parameter[5:])

    mean_counts_ = endemic_term + autoregressive_term + spatiotemporal_term
    mean_counts_backup_ = np.dot(spatial_weight_matrix, mean_counts_)
    priority_features = np.array([mean_counts_, mean_counts_backup_])
    priority_score = expit(np.dot(priority_features, policy_parameter))
    return priority_score


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
    alpha_nu = np.log(model_parameter[0])
    beta_nu = model_parameter[1:2]
    alpha_lambda = np.log(model_parameter[3])
    lambda_a = model_parameter[4]
    alpha_phi = np.log(model_parameter[5])
    phi_a = model_parameter[6]

    env = PoissonDisease(L, lambda_a = lambda_a, phi_a = phi_a, alpha_nu = alpha_nu, alpha_lambda = alpha_lambda,
                         alpha_phi = alpha_phi, beta_nu = beta_nu, Y_initial=Y_current, t_initial=t_current)
    return env


def rollout(Y_current, t_current, policy_parameter, model_parameter, budget, L, time_horizon, discount_factor=0.96):
    total_utility = 0.
    rollout_env = env_from_model_parameter(model_parameter, Y_current, t_current, L)
    rollout_env.reset()

    A = np.zeros(rollout_env.L) # Initial action
    for t in range(time_horizon):
        rollout_env.step(A)
        total_utility -= discount_factor**t * rollout_env.Y.mean()
        X = rollout_env.X
        A = priority_score_policy(policy_parameter, model_parameter, budget, X, rollout_env.spatial_weight_matrix)

    return total_utility


def policy_search(Y_current, t_current, model_parameter, budget, L, time_horizon, discount_factor, policy_optimizer):
    rollout_partial = partial(rollout, Y_current=Y_current, t_current=t_current,
                              model_parameter=model_parameter, budget=budget, L=L, time_horizon=time_horizon,
                              discount_factor=discount_factor)
    policy_parameter_estimate = policy_optimizer(rollout_partial)
    return policy_parameter_estimate


def policy_search_policy(env, budget, time_horizon, discount_factor, policy_optimizer):
    model_parameter_estimate = fit_model(env)
    policy_parameter_estimate = policy_search(env.Y, env.t, model_parameter_estimate, budget, env.L, time_horizon,
                                              discount_factor, policy_optimizer)
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
        total_reward += env.Y.mean()
        A = policy_search_policy(env, budget, time_horizon-t, discount_factor, policy_optimizer)
        env.step(A)
    print(total_reward)

