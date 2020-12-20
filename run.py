import argparse
import numpy as np
from environment import PoissonDisease
from policy_factory import policy_factory
import multiprocessing as mp
from functools import partial

DISCOUNT_FACTOR = 0.96
BURN_IN = 10
BURN_IN_POLICY = policy_factory('random')

def run_replicate(replicate_index, env, budget, time_horizon, policy):
    np.random.seed(replicate_index)

    total_utility = 0.
    env.reset()

    # Burn in
    for t in range(BURN_IN):
        A = BURN_IN_POLICY(env, budget, time_horizon-t, discount_factor)
        env.step(A)

    # Deploy policy
    for t in range(time_horizon):
        total_utility += discount_factor**t * env.Y.mean()
        action_info = policy(env, budget, time_horizon-t, discount_factor)
        if 'propensities' in action_info.keys():
            propensities = action_info
        else:
            propensities = None
        env.step(action_info['A'], propensities=propensities)
    return total_utility


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_name', type=str)
    parser.add_argument('--L', type=int)
    parser.add_argument('--time_horizon', type=int)
    parser.add_argument('--budget', type=int)
    parser.add_argument('--num_replicates', type=int)
    args = parser.parse_args()

    L = args.L
    time_horizon = args.time_horizon
    budget = args.budget
    discount_factor = 0.96
    policy_name = args.policy_name
    policy = policy_factory(policy_name)
    num_replicates = args.num_replicates

    env = PoissonDisease(L=L)

    pool = mp.Pool(processes=num_replicates)
    run_replicate_partial = partial(run_replicate, env=env, budget=budget, time_horizon=time_horizon, policy=policy)
    total_utilities = pool.map(run_replicate_partial, range(num_replicates))
    expected_total_utility = np.mean(total_utilities)
    print(f'policy name: {policy_name} expected value {expected_total_utility}')
