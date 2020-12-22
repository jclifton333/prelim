import os
import numpy as np
import yaml
import pandas as pd


if __name__ == "__main__":
    summary_dict = {'policy': [], 'L': [], 'score': []}
    for fname in os.listdir():
        if fname.endswith(".yml"):
            d = yaml.load(open(fname, 'rb'))
            policy = d['policy']
            L = d['L']
            score = d['score']
            summary_dict['policy'].append(policy)
            summary_dict['L'].append(L)
            summary_dict['score'].append(score)
    summary_df = pd.DataFrame.from_dict(summary_dict)
    summary_df.sort_values(by=['L', 'policy'], inplace=True)
    print(summary_df)
