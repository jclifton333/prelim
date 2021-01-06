import numpy as np
from copy import copy
from .fit_relaxation import fit_lp
from .relaxation import lp_max
from bayes_opt import BayesianOptimization
import pdb
from functools import partial


def expected_utility_at_param(param, rollout, model_parameters, n_rollout_per_it):
    expected_utility = 0.
    for it in range(n_rollout_per_it):
        if isinstance(model_parameters, list):
            utility = rollout(param, model_parameters[it])
        else:
            utility = rollout(param, model_parameters)
        expected_utility += utility / n_rollout_per_it
    return expected_utility


def random_policy_optimizer(rollout, model_parameters, policy_parameter_size=3, n_rollout_per_it=30, n_rep=100):
    params = np.random.uniform(low=0., high=5., size=(n_rep, policy_parameter_size))
    scores = np.zeros(n_rep)
    objective = partial(expected_utility_at_param, rollout=rollout, model_parameters=model_parameters,
                        n_rollout_per_it=n_rollout_per_it)
    for ix, p in enumerate(params):
        score_ix = objective(p)
        scores[ix] = score_ix
    best_ix = np.argsort(scores)[-1]
    best_param = params[best_ix]
    best_score = scores[best_ix]
    return best_param, best_score


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


def gp_policy_optimizer(rollout, model_parameters, n_rollout_per_it=10, n_rep=30, initial_theta=None):
    def objective(theta1, theta2, theta3):
        theta = np.array([theta1, theta2, theta3])
        score = expected_utility_at_param(theta, rollout, model_parameters, n_rollout_per_it=n_rollout_per_it)
        return score

    theta_bounds = (0, 5)
    exploration_parameters = np.random.uniform(low=0., high=5., size=(5, 3))

    explore_ = {'theta1': exploration_parameters[:, 0], 'theta2': exploration_parameters[:, 1],
                'theta3': exploration_parameters[:, 2]}
    bounds = {'theta1': theta_bounds, 'theta2': theta_bounds, 'theta3': theta_bounds}
    bo = BayesianOptimization(objective, bounds)
    bo.explore(explore_)
    bo.maximize(init_points=0, n_iter=n_rep, alpha=1e-4)
    best_param = bo.res['max']['max_params']
    theta_hat = np.array([best_param['theta1'], best_param['theta2'], best_param['theta3']])
    return theta_hat


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


def random_q_optimizer(q, L, budget, n_it=1000):
    A = np.zeros(L)
    A[:budget] = 1
    q_best = float('inf')
    A_best = None
    for it in range(n_it):
        np.random.shuffle(A)
        q_it = q(A).sum()
        if q_it < q_best:
            q_best = q_it
            A_best = copy(A)
    return A_best, q_best


def lp_q_optimizer(q, L, budget):
    coef, intercept = fit_lp(q, L, budget)
    A_best = lp_max(coef, intercept, budget)
    q_best = q(A_best)
    return A_best, q_best


