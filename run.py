import matplotlib.pyplot as plt
import argparse
import numpy as np
from environment import PoissonDisease
from policies.policy_factory import policy_factory
import multiprocessing as mp
from functools import partial
import yaml
import os
import datetime

DISCOUNT_FACTOR = 0.96
BURN_IN = 5
BURN_IN_POLICY = policy_factory('random')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def bootstrap(data, reps=1000):
    """
    Helper for computing standard errors of estimated expected values.
    """
    n = len(data)
    resampled_means = []
    mean_ = np.mean(data)
    for _ in range(reps):
        resampled_data = np.random.choice(data, size=n, replace=True)
        resampled_mean = np.mean(resampled_data - mean_)
        resampled_means.append(resampled_mean)
    se = np.std(resampled_means)
    q_upper = np.percentile(resampled_means, 97.5)
    q_lower = np.percentile(resampled_means, 2.5)
    interval = (float(q_lower + mean_), float(q_upper + mean_))
    return se, interval


def run_replicate(replicate_index, env, budget, time_horizon, policy, discount_factor, specified_kernel):
    np.random.seed(replicate_index*1000)

    total_utility = 0.
    env.reset()

    # Burn in
    for t in range(BURN_IN):
        action_info = BURN_IN_POLICY(env, budget, time_horizon-t, discount_factor)
        env.step(action_info['A'])

    # Deploy policy
    for t in range(time_horizon):
        total_utility += discount_factor**t * env.Y.sum()
        action_info = policy(env, budget, time_horizon-t, discount_factor, kernel=specified_kernel)
        env.step(action_info['A'])
    return total_utility


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_name', type=str)
    parser.add_argument('--L', type=int)
    parser.add_argument('--time_horizon', type=int)
    parser.add_argument('--budget', type=int)
    parser.add_argument('--num_replicates', type=int)
    parser.add_argument('--true_kernel', type=str)
    parser.add_argument('--specified_kernel', type=str)
    parser.add_argument('--global_kernel_bandwidth', type=float)
    parser.add_argument('--replicate_batches', type=int)
    args = parser.parse_args()

    L = args.L
    time_horizon = args.time_horizon
    budget = args.budget
    discount_factor = 0.96
    policy_name = args.policy_name
    policy = policy_factory(policy_name)
    num_replicates = args.num_replicates
    replicate_batches = args.replicate_batches
    true_kernel = args.true_kernel
    specified_kernel = args.specified_kernel
    global_kernel_bandwidth = args.global_kernel_bandwidth

    env = PoissonDisease(L=L, kernel=true_kernel, kernel_bandwidth=global_kernel_bandwidth)
    run_replicate_partial = partial(run_replicate, env=env, budget=budget, time_horizon=time_horizon, policy=policy,
                                    discount_factor=DISCOUNT_FACTOR, specified_kernel=specified_kernel)

    if num_replicates > 1:
        replicates_per_batch = int(num_replicates / replicate_batches)
        pool = mp.Pool(processes=num_replicates)
        all_results = []
        for batch in range(replicate_batches):
            batch_results = pool.map(run_replicate_partial, range(batch*replicates_per_batch, (batch+1)*replicates_per_batch))
            all_results += batch_results
        pool.close()
        expected_total_utility = np.mean(all_results)
        standard_error, interval = bootstrap(all_results)
        standard_error = float(standard_error)

        # Save results
        results = {'policy': policy_name, 'L': L, 'score': float(expected_total_utility), 'se': standard_error,
                   'budget': budget, 'true_kernel': true_kernel, 'specified_kernel': specified_kernel,
                   'global_kernel_bandwidth': global_kernel_bandwidth, 'interval': interval}
        base_name = f'L={L}-{policy_name}-{specified_kernel}'
        prefix = os.path.join(THIS_DIR, 'results', base_name)
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        fname = f'{prefix}_{suffix}.yml'
        with open(fname, 'w') as outfile:
            yaml.dump(results, outfile)
        print(f'L: {L} policy name: {policy_name} expected value {expected_total_utility} interval {interval}')
    else:
        expected_total_utility = run_replicate_partial(1)
        print(f'L: {L} policy name: {policy_name} expected value {expected_total_utility}')

