import numpy as np
import policy_search
import model_estimation


def random_policy(env, budget, time_horizon, discount_factor):
    A = np.zeros(env.L)
    A[:budget] = 1
    np.random.shuffle(A)
    random_action_prob = budget / env.L
    propensities = random_action_prob * A + (1 - random_action_prob) * (1 - A)
    return {'A': A, 'propensities': propensities}


def treat_all_policy(env, budget, time_horizon, discount_factor):
    A = np.ones(env.L)
    return {'A': A}


def treat_none_policy(env, budget, time_horizon, discount_factor):
    A = np.zeros(env.L)
    return {'A': A}


def greedy_model_based_policy(env, budget, time_horizon, discount_factor):
    model_parameter_estimate = model_estimation.fit_model(env)
    mean_counts_ = policy_search.mean_counts_from_model_parameter(model_parameter_estimate, env.X)
    A = np.zeros(env.L)
    highest_mean_counts = np.argsort(mean_counts_)[-budget:]
    A[highest_mean_counts] = 1
    return {'A': A}


def oracle_greedy_model_based_policy(env, budget, time_horizon, discount_factor):
    model_parameter = policy_search.model_parameter_from_env(env)
    mean_counts_ = policy_search.mean_counts_from_model_parameter(model_parameter, env.X)
    A = np.zeros(env.L)
    highest_mean_counts = np.argsort(mean_counts_)[-budget:]
    A[highest_mean_counts] = 1
    return {'A': A}


