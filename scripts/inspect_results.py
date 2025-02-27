from typing import Dict

import relax
from pathlib import Path
import re
import csv
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
import numpy as np


sns.set_style('whitegrid')
sns.set_context(font_scale=2.6)


def plot_mean(patterns_dict: Dict, env_name, fig_name = None,
              max_steps=None):
    plt.figure(figsize=(4*1.25, 3*1.25))
    package_path = Path(relax.__file__)
    logdir = package_path.parent.parent / 'logs' / env_name
    dfs = []
    for alg, pattern in patterns_dict.items():
        matching_dir = [s for s in logdir.iterdir() if re.match(pattern, str(s))]
        for dir in matching_dir:
            csv_path = dir / 'log.csv'
            df = pd.read_csv(str(csv_path))
            print(str(dir), "dir")
            df.loc[:, ('seed')] = str(dir).split('_s')[1].split('_')[0]
            df.loc[:, ('alg')] = alg
            dfs.append(df)
    try:
        total_df = pd.concat(dfs, ignore_index=True)
    except:
        return
    # if max_steps is not None:
    plt.ylim(-0.1, 1.1)
    plt.title(env_name.split("/")[1])
    sns.lineplot(data=total_df, x='step', y='success_rate', hue='alg')
    if fig_name is not None:
        plt.savefig(fig_name)
    else:
        plt.show()
    


def load_best_results(pattern, env_name, show_df=False,
              max_steps=None):
    package_path = Path(relax.__file__)
    logdir = package_path.parent.parent / 'logs' / env_name
    
    matching_dir = [s for s in logdir.iterdir() if re.match(pattern, str(s))]
    dfs = []
    for dir in matching_dir:
        csv_path = dir / 'log.csv'
        df = pd.read_csv(str(csv_path))
        if max_steps is not None:
            df = df[df['step'] < max_steps]
        sliced_df = df.loc[df['avg_ret'].idxmax()]
        sliced_df.loc['seed'] = str(dir).split('_s')[1].split('_')[0]
        dfs.append(sliced_df)
    total_df = pd.concat(dfs, ignore_index=True, axis=1).T
    if show_df:
        print(total_df.to_markdown())
    print(f"${total_df['avg_ret'].mean():.0f} \pm {total_df['avg_ret'].std():.0f}$")
    return total_df

if __name__ == "__main__":
    for env in ['disassemble-v2-goal-observable', 'bin-picking-v2-goal-observable']:
        patterns_dict = {
            'rep_weight_0': r".*/diffrep.*random_seed_rep_test_weight_mu_update_0.0.*",
            'rep_weight_0.01': r".*/diffrep.*random_seed_rep_test_weight_mu_update_0.01.*",
            'rep_weight_10': r".*/diffrep.*random_seed_rep_test_weight_mu_update_10.*",
        }
        plot_mean(patterns_dict, f'metaworld/{env}', f"figures/metaworld_{env}.pdf")