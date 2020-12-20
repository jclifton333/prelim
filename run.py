import argparse
import numpy as np
from environment import PoissonDisease
from policy_factory import policy_factory
import multiprocessing as mp
from functools import partial
import yaml
import os
import datetime

DISCOUNT_FACTOR = 0.96
BURN_IN = 5
BURN_IN_POLICY = policy_factory('random')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def run_replicate(replicate_index, env, budget, time_horizon, policy, discount_factor):
    np.random.seed(replicate_index)

    total_utility = 0.
    env.reset()

    # Burn in
    for t in range(BURN_IN):
        action_info = BURN_IN_POLICY(env, budget, time_horizon-t, discount_factor)
        env.step(action_info['A'], propensities=action_info['propensities'])

    # Deploy policy
    for t in range(time_horizon):
        total_utility += discount_factor**t * env.Y.mean()
        action_info = policy(env, budget, time_horizon-t, discount_factor)
        if 'propensities' in action_info.keys():
            propensities = action_info['propensities']
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
    run_replicate_partial = partial(run_replicate, env=env, budget=budget, time_horizon=time_horizon, policy=policy,
                                    discount_factor=DISCOUNT_FACTOR)

    if num_replicates > 1:
        pool = mp.Pool(processes=num_replicates)
        total_utilities = pool.map(run_replicate_partial, range(num_replicates))
        expected_total_utility = np.mean(total_utilities)
        standard_error = np.std(total_utilities) / np.sqrt(num_replicates)
    else:
        expected_total_utility = run_replicate_partial(1)
        standard_error = None

    # Display and save results
    print(f'L: {L} policy name: {policy_name} expected value {expected_total_utility}')
    results = {'policy': policy_name, 'L': L, 'score': float(expected_total_utility), 'se': float(standard_error),
               'budget': budget}
    base_name = f'L={L}-{policy_name}'
    prefix = os.path.join(THIS_DIR, 'results', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    fname = f'{prefix}_{suffix}.yml'
    with open(fname, 'w') as outfile:
      yaml.dump(results, outfile)
