import numpy as np
from sklearn.ensemble import RandomForestRegressor
import optim


def myopic_model_free_policy(env, budget, time_horizon, discount_factor, q_optimizer=optim.random_q_optimizer):
    X = np.vstack(env.X_list)
    Y = np.vstack(env.Y_list)
    rf = RandomForestRegressor()
    rf.fit(X, Y)

    X_current = env.X

    def q(A_):
       X_at_A = env.get_X_at_A(X_current, A_)
       q_ = rf.predict(X_at_A).sum()
       return q_

    A = q_optimizer(q, env.L, budget)
    return A




