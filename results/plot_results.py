import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdb


if __name__ == "__main__":
  policy_rename_map = {'mbm-greedy_model_based,myopic_model_based': 'mbm'}
  df = pd.read_csv('210718,210719.csv')

  # Clean df for plotting
  df['correctly_specified'] = df['specified_kernel'] == 'global'
  df['policy'] = df['policy'].replace(policy_rename_map)

  grouped = df.groupby(['policy', 'bandwidth', 'correctly_specified'])
  summary_to_plot = grouped.describe()[[('score', 'mean'), ('lower', 'mean'), ('upper', 'mean')]].reset_index()
  summary_to_plot['err'] = (summary_to_plot[('upper', 'mean')] - summary_to_plot[('lower', 'mean')]) / 2
  # summary_to_plot = grouped.score.agg('mean').reset_index()

  def errplot(x, y, yerr, **kwargs):
    ax = plt.gca()
    data = kwargs.pop('data')
    data.plot(x=x, y=y, yerr=yerr, kind='scatter', ax=ax, **kwargs)

  g = sns.FacetGrid(summary_to_plot, col='bandwidth', hue='correctly_specified')
  g.map_dataframe(errplot, 'policy', 'score', 'err')
  g.add_legend()
  # plt.legend(loc=1, title='correctly specified')
  plt.show()
  # Plot
  # sns.catplot(y='score', col='bandwidth', x='policy', kind='point', data=df)
  # plt.show()
