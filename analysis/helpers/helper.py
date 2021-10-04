"""Helpers function to analyse experiment data."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_data(experiment_name: str) -> pd.DataFrame:
    """Read the batch data from a pickle."""
    return pd.read_pickle(f'experiments/{experiment_name}/batch_data.p')


def count_explored_tiles_agents(env_df: pd.DataFrame) -> dict:
    """Count how many tiles a agent has explored per tik."""
    dict_agent_info = dict()
    for agent in range(env_df['agents_n'].max()):
        list_agent_info = []
        for i in range(len(env_df)):
            list_agent_info.append(env_df.iloc[i]['explored'][agent].sum())
        dict_agent_info[agent] = list_agent_info
    return dict_agent_info


def plot_batch(env_df: pd.DataFrame) -> None:
    """Plot result per batch."""
    for batch in range(len(env_df)):
        sub_df = pd.DataFrame.from_dict(env_df[batch]).set_index('time')
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        fig.suptitle('Explored maze by agents')
        plt.xlabel("time step")
        plt.ylabel("explored tiles")
        sns.lineplot(axes=axes, data=count_explored_tiles_agents(sub_df))
        plt.show()
