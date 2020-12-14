import argparse
import numpy as np
from environment import PoissonDisease
from policy_factory import policy_factory

DISCOUNT_FACTOR = 0.96
BURN_IN = 5
BURN_IN_POLICY = policy_factory('random')

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
    replicate_total_utilities = np.zeros(num_replicates)

    for replicate in range(num_replicates):
        env.reset()

        # Burn in
        for t in range(BURN_IN):
            A = BURN_IN_POLICY(env, budget, time_horizon-t, discount_factor)
            env.step(A)

        # Deploy policy
        for t in range(time_horizon):
            replicate_total_utilities[replicate] += discount_factor**t * env.Y.mean()
            A = policy(env, budget, time_horizon-t, discount_factor)
            env.step(A)
    expected_total_utility = replicate_total_utilities.mean()
    print(f'policy name: {policy_name} expected value {expected_total_utility}')
