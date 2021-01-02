import numpy as np
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from scipy.stats import norm
from .policy_search import mean_counts_from_model_parameter, model_parameter_from_env
from .model_estimation import fit_model
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


def myopic_model_based_policy(env, budget, time_horizon, discount_factor, q_optimizer=optim.lp_q_optimizer,
                              kernel='network'):
    model_parameter_estimate = fit_model(env)
    X_current = env.X
    K_current = env.get_current_K(kernel)

    def q(A_):
        X_at_A, K_at_A = env.get_X_at_A(X_current, K_current, A_, kernel=kernel)
        q_ = mean_counts_from_model_parameter(model_parameter_estimate, X_at_A, K_at_A)
        return q_

    A, q_best = q_optimizer(q, env.L, budget)
    return {'A': A}


def oracle_myopic_model_based_policy(env, budget, time_horizon, discount_factor, q_optimizer=optim.lp_q_optimizer):

    model_parameter = model_parameter_from_env(env)
    X_current = env.X
    K_current = env.get_current_K(kernel='true')

    def q(A_):
        X_at_A, K_at_A = env.get_X_at_A(X_current, K_current, A_, kernel='true')
        q_ = mean_counts_from_model_parameter(model_parameter, X_at_A, K_at_A)
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


def oracle_one_step_fitted_q(env, budget, time_horizon, discount_factor, q_optimizer=optim.lp_q_optimizer,
                             backup_regressor=RandomForestRegressor, **kwargs):

    model_parameter = model_parameter_from_env(env)
    K_list = env.get_K_history(kernel='true')

    def q0(A_, X_, K_):
        if A_ is not None:
            X_at_A, K_at_A = env.get_X_at_A(X_, K_, A_, kernel=None)
        else:
            X_at_A = X_
            K_at_A = K_
        q_ = mean_counts_from_model_parameter(model_parameter, X_at_A, K_at_A)
        return q_

    # Get backed-up q values using myopic model-based
    r = np.zeros(0)
    v1 = np.zeros(0)
    for t in range(len(env.X_list)-1):
        X_t = env.X_list[t]
        X_tp1 = env.X_list[t+1]
        K_t = K_list[t]
        K_tp1 = K_list[t+1]

        # Immediate reward
        r_t = q0(None, X_t, K_t)
        r = np.hstack((r, r_t))

        # Next step reward
        q_at_Xtp1 = partial(q0, X_=X_tp1, K_=K_tp1)
        A_opt, _ = q_optimizer(q_at_Xtp1, env.L, budget)
        v1_t = q_at_Xtp1(A_opt)
        v1 = np.hstack((v1, v1_t))

    # Distill
    X0 = np.vstack(env.X_list[:-1])
    K0 = np.vstack(K_list[:-1])
    X0 = np.column_stack((X0, K0))
    target = r + discount_factor*v1
    model1 = backup_regressor()
    model1.fit(X0, target)

    # Evaluate policy at current state
    X_current = env.X
    K_current = env.get_current_K(kernel='true')

    def q1(A_):
        X_at_A, K_at_A = env.get_X_at_A(X_current, K_current, A_)
        X_at_A = np.column_stack((X_at_A, K_at_A))
        q_ = model1.predict(X_at_A)
        return q_

    A, _ = q_optimizer(q1, env.L, budget)
    return {'A': A}


def one_step_fitted_q_policy(env, budget, time_horizon, discount_factor, q_optimizer=optim.lp_q_optimizer,
                             regressor=RandomForestRegressor, backup_regressor=RandomForestRegressor,
                             kernel='network'):
    X = np.vstack(env.X_list)
    Y = np.hstack(env.Y_list)
    K_list = env.get_K_history(kernel)
    K = np.vstack(K_list)
    X = np.column_stack((X, K))

    # Fit zero-step q
    model0 = regressor()
    model0.fit(X, Y)

    def q0(A_, X_, K_):
        X_at_A, K_at_A = env.get_X_at_A(X_, K_, A_, kernel=kernel)
        X_at_A = np.column_stack((X_at_A, K_at_A))
        q_ = model0.predict(X_at_A)
        return q_

    # Get backed-up q values
    v1 = np.zeros(0)
    for X_, K_ in zip(env.X_list[1:], K_list[1:]):
        q_at_X = partial(q0, X_=X_, K_=K_)
        A_best, _ = q_optimizer(q_at_X, env.L, budget)
        X_at_A_best, K_at_A_best = env.get_X_at_A(X_, K_, A_best)
        X_at_A_best = np.column_stack((X_at_A_best, K_at_A_best))
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
    K_current = env.get_current_K(kernel)

    def q1(A_):
        X_at_A, K_at_A = env.get_X_at_A(X_current, K_current, A_)
        X_at_A = np.column_stack((X_at_A, K_at_A))
        q_ = model1.predict(X_at_A)
        return q_

    A, _ = q_optimizer(q1, env.L, budget)
    return {'A': A}








