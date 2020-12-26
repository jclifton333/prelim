import os
import numpy as np
import yaml
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str)
    args = parser.parse_args()

    date = args.date

    summary_dict = {'policy': [], 'L': [], 'score': [], 'se': []}
    for fname in os.listdir():
        if fname.endswith(".yml"):
            if date is None or date in fname:
                d = yaml.load(open(fname, 'rb'))
                policy = d['policy']
                L = d['L']
                score = d['score']
                se = d['se']
                summary_dict['policy'].append(policy)
                summary_dict['L'].append(L)
                summary_dict['score'].append(score)
                summary_dict['se'].append(se)
    summary_df = pd.DataFrame.from_dict(summary_dict)
    summary_df.sort_values(by=['L', 'policy'], inplace=True)
    print(summary_df)
