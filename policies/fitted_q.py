import numpy as np
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from scipy.stats import norm
import prelim.optim.optim as optim
import pdb


def myopic_model_free_policy(env, budget, time_horizon, discount_factor, q_optimizer=optim.lp_q_optimizer,
                             regressor=Ridge, kernel='network'):
    X = np.vstack(env.X_list)
    K_list = env.get_K_history(kernel)
    K = np.vstack(K_list)
    X = np.column_stack((X, K))
    Y = np.hstack(env.Y_list)
    model = regressor()
    model.fit(X, Y)

    X_current = env.X
    K_current = env.get_current_K(kernel)

    def q(A_):
        X_at_A, K_at_A = env.get_X_at_A(X_current, K_current, A_, kernel=kernel)
        X_at_A = np.column_stack((X_at_A, K_at_A))
        q_ = model.predict(X_at_A)
        return q_

    A, q_best = q_optimizer(q, env.L, budget)
    return {'A': A}


def greedy_model_free_policy(env, budget, time_horizon, discount_factor, regressor=Ridge, kernel='network'):
    X = np.vstack(env.X_list)
    K_list = env.get_K_history(kernel)
    K = np.vstack(K_list)
    X = np.column_stack((X, K))
    Y = np.hstack(env.Y_list)
    model = regressor()
    model.fit(X, Y)

    X_current = env.X
    K_current = env.get_current_K(kernel)
    mean_counts_ = model.predict(np.column_stack((X_current, K_current)))
    highest_mean_counts = np.argsort(mean_counts_)[-budget:]
    A = np.zeros(env.L)
    A[highest_mean_counts] = 1
    return {'A': A}


def epsilon_greedy_propensity_score_from_A(A, spatial_weight_matrix, epsilon, prob_random_action,
                                           spillover_propensity_means, spillover_propensity_variances):
    A_spillover = np.dot(spatial_weight_matrix, A)
    A_spillover_propensities = np.array([norm.pdf(a, loc=m, scale=s)
                                         for a, m, s in zip(A_spillover,
                                                            spillover_propensity_means,
                                                            spillover_propensity_variances)]) * epsilon
    A_propensities = epsilon * (A * prob_random_action + (1 - A) * (1 - prob_random_action)) + \
                     (1 - epsilon) * A
    propensities = np.multiply(A_propensities, A_spillover_propensities)
    return propensities


def one_step_fitted_q_policy(env, budget, time_horizon, discount_factor, q_optimizer=optim.random_q_optimizer,
                             regressor=Ridge, backup_regressor=RandomForestRegressor, epsilon=0.1):
    # ToDo: incorporate kernel features

    X = np.vstack(env.X_list)
    Y = np.hstack(env.Y_list)

    # Fit zero-step q
    model0 = regressor()
    model0.fit(X, Y)

    def q0(A_, X_):
        X_at_A = env.get_X_at_A(X_, A_)
        q_ = model0.predict(X_at_A).sum()
        return q_

    # Get backed-up q values
    v1 = np.zeros(0)
    for X_ in env.X_list[1:]:
        q_at_X = partial(q0, X_=X_)
        A_best, _ = q_optimizer(q_at_X, env.L, budget)
        X_at_A_best = env.get_X_at_A(X_, A_best)
        v = model0.predict(X_at_A_best)
        v1 = np.hstack((v1, v))

    # Estimate q1
    X0 = X[:-env.L, :]
    X1 = X[env.L:, :]
    rhat = model0.predict(X1)
    backup = rhat + discount_factor * v1
    model1 = backup_regressor()
    model1.fit(X0, backup)

    # Minimize q1 estimate
    X_current = env.X

    def q1(A_):
        X_at_A = env.get_X_at_A(X_current, A_)
        q_ = model1.predict(X_at_A).sum()
        return q_

    A_opt, _ = q_optimizer(q1, env.L, budget)

    # epsilon-greedy exploration
    if np.random.uniform() < epsilon:
        A = np.zeros(env.L)
        A[:budget] = 1
        np.random.shuffle(A)
    else:
        A = A_opt

    return {'A': A}


def one_step_fitted_q_propensity_policy(env, budget, time_horizon, discount_factor, q_optimizer=optim.random_q_optimizer,
                                        regressor=Ridge, backup_regressor=RandomForestRegressor, epsilon=0.1):
    X = np.vstack(env.X_list)
    Y = np.hstack(env.Y_list)

    # Fit zero-step q
    model0 = regressor()
    model0.fit(X, Y)

    def q0(A_, X_):
        X_at_A = env.get_X_at_A(X_, A_)
        q_ = model0.predict(X_at_A).sum()
        return q_

    # Get backed-up q values
    v1 = np.zeros(0)
    for X_ in env.X_list[1:]:
        q_at_X = partial(q0, X_=X_)
        A_best, _ = q_optimizer(q_at_X, env.L, budget)
        X_at_A_best = env.get_X_at_A(X_, A_best)
        v = model0.predict(X_at_A_best)
        v1 = np.hstack((v1, v))

    # Estimate q1
    X0 = X[:-env.L, :]
    P = np.hstack(env.propensities_list[:-1])
    X0 = np.column_stack((X0, P))
    X1 = X[env.L:, :]
    rhat = model0.predict(X1)
    backup = rhat + discount_factor*v1
    model1 = backup_regressor()
    model1.fit(X0, backup)

    # Minimize q1 estimate
    X_current = env.X
    prob_random_action = budget / env.L
    spillover_propensity_means = env.spatial_weight_matrix.sum(axis=1) * prob_random_action
    spillover_propensity_variances = (env.spatial_weight_matrix ** 2).sum(axis=1) * prob_random_action * \
                                     (1 - prob_random_action)

    def q1(A_):
        X_at_A = env.get_X_at_A(X_current, A_)
        propensities = epsilon_greedy_propensity_score_from_A(A_, env.spatial_weight_matrix, epsilon,
                                                              prob_random_action, spillover_propensity_means,
                                                              spillover_propensity_variances)
        X_at_A = np.column_stack((X_at_A, propensities))
        q_ = model1.predict(X_at_A).sum()
        return q_

    A_opt, _ = q_optimizer(q1, env.L, budget)

    # epsilon-greedy exploration
    if np.random.uniform() < epsilon:
        A = np.zeros(env.L)
        A[:budget] = 1
        np.random.shuffle(A)
    else:
        A = A_opt

    # ToDo: double check propensity calculations
    # ToDo: Propensity should be P( A = 1 | x ), not P( A = a_obs | X ) ?
    propensities = epsilon_greedy_propensity_score_from_A(A, env.spatial_weight_matrix, epsilon, prob_random_action,
                                                          spillover_propensity_means, spillover_propensity_variances)

    return {'A': A, 'propensities': propensities}






