import numpy as np
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import optim


def myopic_model_free_policy(env, budget, time_horizon, discount_factor, q_optimizer=optim.random_q_optimizer,
                             regressor=Ridge):
    X = np.vstack(env.X_list)
    Y = np.hstack(env.Y_list)
    model = regressor()
    model.fit(X, Y)

    X_current = env.X

    def q(A_):
       X_at_A = env.get_X_at_A(X_current, A_)
       q_ = model.predict(X_at_A).sum()
       return q_

    A, q_best = q_optimizer(q, env.L, budget)
    return A


def one_step_fitted_q_policy(env, budget, time_horizon, discount_factor, q_optimizer=optim.random_q_optimizer,
                             regressor=Ridge, backup_regressor=RandomForestRegressor):
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
    X1 = X[env.L:, :]
    rhat = model0.predict(X1)
    backup = rhat + discount_factor*v1
    model1 = backup_regressor()
    model1.fit(X1, backup)

    # Minimize q1 estimate
    X_current = env.X

    def q1(A_):
        X_at_A = env.get_X_at_A(X_current, A_)
        q_ = model1.predict(X_at_A).sum()
        return q_

    A, _ = q_optimizer(q1, env.L, budget)
    return A







