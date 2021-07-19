import os
import numpy as np
import yaml
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str)
    parser.add_argument('--bandwidth_filter', type=str)
    parser.add_argument('--policy_filter', type=str)
    parser.add_argument('--save', type=int, default=0)
    args = parser.parse_args()

    date = args.date
    date_lst = date.split(',')
    if args.bandwidth_filter is not None:
        bandwidth_filter = [float(b) for b in args.bandwidth_filter.split(',')]
    else:
        bandwidth_filter = None

    if args.policy_filter is not None:
        policy_filter = [b for b in args.policy_filter.split(',')]
    else:
        policy_filter = None

    summary_dict = {'policy': [], 'L': [], 'score': [], 'lower': [], 'upper': [], 'specified_kernel': [],
                    'bandwidth': []}
    for fname in os.listdir():
        if fname.endswith(".yml"):
            matches_date = False
            for date in date_lst:
                if date in fname:
                    matches_date = True
                    break
            if date is None or matches_date:
                d = yaml.load(open(fname, 'rb'))
                policy = d['policy']
                L = d['L']
                score = d['score']
                specified_kernel = d['specified_kernel']
                if 'interval' in d.keys():
                  interval = d['interval']
                  lower, upper = interval
                else:
                  lower, upper = None, None
                if 'global_kernel_bandwidth' in d.keys():
                    bandwidth = d['global_kernel_bandwidth']
                else:
                    bandwidth = None

                policy = d['policy']

                # Check filters
                if bandwidth_filter is not None:
                    keep = bandwidth in bandwidth_filter
                else: 
                    keep = True
                if keep:
                    if policy_filter is not None:
                      keep = policy in policy_filter

                if keep:
                    summary_dict['specified_kernel'].append(specified_kernel)
                    summary_dict['bandwidth'].append(bandwidth)
                    summary_dict['policy'].append(policy)
                    summary_dict['L'].append(L)
                    summary_dict['score'].append(score)
                    summary_dict['lower'].append(lower)
                    summary_dict['upper'].append(upper)

    summary_df = pd.DataFrame.from_dict(summary_dict)
    summary_df.sort_values(by=['L', 'policy', 'specified_kernel'], inplace=True)
    summary_df = summary_df[['L', 'policy', 'specified_kernel', 'bandwidth', 'score', 'lower', 'upper']]
    if args.save:
        summary_df.to_csv(f'{date}.csv')
    print(summary_df)
