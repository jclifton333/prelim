import numpy as np
from scipy.special import expit
from environment import PoissonDisease
from model_estimation import fit_model
from functools import partial
import copy
import optim
from numba import njit
import pdb


def priority_scores(policy_parameter, model_parameter, X, spatial_weight_matrix):
    mean_counts_ = mean_counts_from_model_parameter(model_parameter, X)
    Y = X[:, 3]
    spatial_weight_times_ytm1 = X[:, 5]
    priority_features = np.column_stack([mean_counts_, Y, spatial_weight_times_ytm1])
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

    mean_counts_ = endemic_term + autoregressive_term + spatiotemporal_term
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


def model_parameter_from_env(env):
    model_parameter = np.zeros(7)
    model_parameter[0] = env.alpha_nu
    model_parameter[1:3] = np.log(env.beta_nu)
    model_parameter[3] = env.alpha_lambda
    model_parameter[4] = np.log(env.lambda_a)
    model_parameter[5] = env.alpha_phi
    model_parameter[6] = np.log(env.phi_a)
    return model_parameter


def env_from_model_parameter(model_parameter, Y_current, t_current, L):
    alpha_nu = model_parameter[0]
    beta_nu = np.exp(model_parameter[1:3])
    alpha_lambda = model_parameter[3]
    lambda_a = np.exp(model_parameter[4])
    alpha_phi = model_parameter[5]
    phi_a = np.exp(model_parameter[6])

    env = PoissonDisease(L, lambda_a = lambda_a, phi_a = phi_a, alpha_nu = alpha_nu, alpha_lambda = alpha_lambda,
                         alpha_phi = alpha_phi, beta_nu = beta_nu, Y_initial=Y_current, t_initial=t_current)
    return env


def rollout(policy_parameter, env, budget, time_horizon, model_parameter=None, discount_factor=0.96, oracle=False):
    total_utility = 0.
    if oracle:
        rollout_env = copy.deepcopy(env)
    else:
        model_parameter = fit_model(env, perturb=True)
        rollout_env = env_from_model_parameter(model_parameter, env.Y, env.t, env.L)
    rollout_env.reset()

    A = np.zeros(rollout_env.L)  # Initial action
    for t in range(time_horizon):
        rollout_env.step(A)
        total_utility -= discount_factor**t * rollout_env.Y.mean()
        X = rollout_env.X
        A = priority_score_policy(policy_parameter, model_parameter, budget, X, rollout_env.spatial_weight_matrix)

    return total_utility


def policy_search(env, budget, time_horizon, discount_factor, policy_optimizer, oracle=False,
                  model_parameter=None):
    # ToDo: can be optimized by first getting list of bootstrap replicates, rather than re-doing for every candidate policy
    rollout_partial = partial(rollout, env=env, budget=budget, time_horizon=time_horizon,
                              discount_factor=discount_factor, oracle=oracle, model_parameter=model_parameter)
    policy_parameter_estimate = policy_optimizer(rollout_partial)
    return policy_parameter_estimate


def policy_search_policy(env, budget, time_horizon, discount_factor,
                         policy_optimizer=optim.genetic_policy_optimizer):
    model_parameter_estimate = fit_model(env, perturb=False)
    policy_parameter_estimate = policy_search(env, budget, time_horizon, discount_factor, policy_optimizer)
    A = priority_score_policy(policy_parameter_estimate, model_parameter_estimate, budget, env.X,
                              env.spatial_weight_matrix)
    return {'A': A}


def oracle_policy_search_policy(env, budget, time_horizon, discount_factor,
                                policy_optimizer=optim.genetic_policy_optimizer):
    model_parameter = model_parameter_from_env(env)
    policy_parameter_estimate = policy_search(env, budget, time_horizon, discount_factor, policy_optimizer,
                                              oracle=True, model_parameter=model_parameter)
    A = priority_score_policy(policy_parameter_estimate, model_parameter, budget, env.X,
                              env.spatial_weight_matrix)
    return {'A': A}


if __name__ == "__main__":
    # ToDo: replace bootstrap distribution by asymptotic distribution

    L = 50
    time_horizon = 10
    budget = 10
    discount_factor = 0.96
    policy_optimizer = optim.genetic_policy_optimizer

    env = PoissonDisease(L=L)
    env.reset()

    A = np.zeros(L)
    env.step(A)
    total_reward = 0.
    for t in range(time_horizon):
        total_reward += discount_factor**t * env.Y.mean()
        action_info = oracle_policy_search_policy(env, budget, time_horizon-t, discount_factor, policy_optimizer)
        env.step(action_info['A'])
        print(t, total_reward)

