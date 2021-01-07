import numpy as np
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from scipy.stats import norm
from .policy_search import mean_counts_from_model_parameter, model_parameter_from_env
from .model_estimation import fit_model
import prelim.optim.optim as optim


def myopic_model_free_policy(env, budget, time_horizon, discount_factor, q_optimizer=optim.lp_q_optimizer,
                             regressor=Ridge, kernel='network'):

    # Estimate mean local infection count given current local features
    X = np.vstack(env.X_list)
    K_list = env.get_K_history(kernel)
    K = np.vstack(K_list)
    X = np.column_stack((X, K))
    Y = np.hstack(env.Y_list)
    model = regressor()
    model.fit(X, Y)

    # Predict conditional rewards at current state, different actions A_
    X_current = env.X
    K_current = env.get_current_K(kernel)

    def q(A_):
        X_at_A, K_at_A = env.get_features_at_action(X_current, K_current, A_, kernel=kernel)
        X_at_A = np.column_stack((X_at_A, K_at_A))
        q_ = model.predict(X_at_A)
        return q_

    # Optimize estimated conditional reward
    A, q_best = q_optimizer(q, env.L, budget)
    return {'A': A}


def myopic_model_based_policy(env, budget, time_horizon, discount_factor, q_optimizer=optim.lp_q_optimizer,
                              kernel='network'):

    # Evaluate estimated infection model at current state and different actions A_
    model_parameter_estimate = fit_model(env)
    X_current = env.X
    K_current = env.get_current_K(kernel)

    def q(A_):
        X_at_A, K_at_A = env.get_features_at_action(X_current, K_current, A_, kernel=kernel)
        q_ = mean_counts_from_model_parameter(model_parameter_estimate, X_at_A, K_at_A)
        return q_

    # Optimize estimated model-based conditional reward
    A, q_best = q_optimizer(q, env.L, budget)
    return {'A': A}


def oracle_myopic_model_based_policy(env, budget, time_horizon, discount_factor, q_optimizer=optim.lp_q_optimizer,
                                     kernel='true'):

    # Evaluate true infection model at current state and different actions A_
    model_parameter = model_parameter_from_env(env)
    X_current = env.X
    K_current = env.get_current_K(kernel=kernel)

    def q(A_):
        X_at_A, K_at_A = env.get_features_at_action(X_current, K_current, A_, kernel=kernel)
        q_ = mean_counts_from_model_parameter(model_parameter, X_at_A, K_at_A)
        return q_

    # Optimize expected reward
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


def oracle_one_step_fitted_q(env, budget, time_horizon, discount_factor, q_optimizer=optim.random_q_optimizer,
                             num_mc_reps=10, **kwargs):

    model_parameter = model_parameter_from_env(env)

    def q0(A_, X_, K_):
        if A_ is not None:
            X_at_A, K_at_A = env.get_features_at_action(X_, K_, A_, kernel=None)
        else:
            X_at_A = X_
            K_at_A = K_
        q_ = mean_counts_from_model_parameter(model_parameter, X_at_A, K_at_A)
        return q_

    # Get backed-up q values using myopic model-based
    X_current = env.X
    K_current = env.get_current_K(kernel='true')

    # Query policy at current state
    def q1(A_):
        q1_ = np.zeros(env.L)
        X0_list = []

        # Monte Carlo estimate of max_A' E[ q_0(Stp, A') | X_current, A_]
        for _ in range(num_mc_reps):
            X_at_A_dummy, K_at_A_dummy = env.get_features_at_action(X_current, K_current, A_, kernel=None)
            X0_list.append(np.column_stack((X_at_A_dummy, K_at_A_dummy)))
            Xtp, Ktp = env.draw_next_state(A_)
            r_rep = Xtp[:, 3]

            # Next step reward
            q_at_Xtp1 = partial(q0, X_=Xtp, K_=Ktp)
            A_opt, _ = q_optimizer(q_at_Xtp1, env.L, budget)
            rp1_rep = q_at_Xtp1(A_opt)

            q1_rep = r_rep + discount_factor * rp1_rep
            q1_ += q1_rep / num_mc_reps
        return q1_

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
        X_at_A, K_at_A = env.get_features_at_action(X_, K_, A_, kernel=kernel)
        X_at_A = np.column_stack((X_at_A, K_at_A))
        q_ = model0.predict(X_at_A)
        return q_

    # Get backed-up q values
    v1 = np.zeros(0)
    for X_, K_ in zip(env.X_list[1:], K_list[1:]):
        q_at_X = partial(q0, X_=X_, K_=K_)
        A_best, _ = q_optimizer(q_at_X, env.L, budget)
        X_at_A_best, K_at_A_best = env.get_features_at_action(X_, K_, A_best)
        X_at_A_best = np.column_stack((X_at_A_best, K_at_A_best))
        v = model0.predict(X_at_A_best)
        v1 = np.hstack((v1, v))

    # Estimate q1
    X0 = X[:-env.L, :]
    rhat = model0.predict(X0)
    backup = rhat + discount_factor * v1
    model1 = backup_regressor()
    model1.fit(X0, backup)

    # Minimize q1 estimate
    X_current = env.X
    K_current = env.get_current_K(kernel)

    def q1(A_):
        X_at_A, K_at_A = env.get_features_at_action(X_current, K_current, A_)
        X_at_A = np.column_stack((X_at_A, K_at_A))
        q_ = model1.predict(X_at_A)
        return q_

    A, _ = q_optimizer(q1, env.L, budget)
    return {'A': A}








