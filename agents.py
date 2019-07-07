# ========================================================================
#
# Imports
#
# ========================================================================
import os
import numpy as np
import pandas as pd
import utilities


# ========================================================================
#
# Classes
#
# ========================================================================
class Agent:
    def __init__(self, env):
        self.env = env


# ========================================================================
class CalibratedAgent(Agent):
    def __init__(self, env):
        Agent.__init__(self, env)

    def learn(self):
        self.actions = utilities.interpolate_df(
            self.env.envs[0].history.ca,
            "ca",
            pd.read_csv(os.path.join("datafiles", "calibrated_data.csv")),
        )
        self.actions.index = self.env.envs[0].history.index

    def predict(self, state):

        # Find the action matching the current CA
        idx = (
            np.abs(
                self.env.envs[0].history.ca
                - state.flatten()[self.env.envs[0].observables.index("ca")]
            )
        ).idxmin()

        return [self.actions.loc[idx + 1, ["mdot", "qdot"]].values], {}

    def save(self, name):
        # do nothing
        return 0

    def load(self, name):
        self.learn()
        return 0

    def generate_expert_traj(self, fname):
        df, total_reward = utilities.evaluate_agent(self.env, self)
        episode_starts = [False for _ in range(len(df))]
        episode_starts[0] = True
        episode_starts[-1] = True

        numpy_dict = {
            "actions": df[["mdot", "qdot"]].values,
            "obs": df[self.env.get_attr("observables", indices=0)[0]].values,
            "rewards": df.rewards.values,
            "episode_returns": np.array([total_reward]),
            "episode_starts": episode_starts,
        }
        np.savez(fname, **numpy_dict)
