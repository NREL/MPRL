# ========================================================================
#
# Imports
#
# ========================================================================
import os
import numpy as np
import pandas as pd
import mprl.utilities as utilities


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
        self.datadir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "datafiles"
        )

    def learn(self):
        self.actions = utilities.interpolate_df(
            self.env.envs[0].history.ca,
            "ca",
            pd.read_csv(os.path.join(self.datadir, "calibrated_data.csv")),
        )
        self.actions.index = self.env.envs[0].history.index
        self.actions.mdot[self.actions.mdot < 0] = 0
        self.actions = self.actions[self.env.envs[0].action.actions]

    def predict(self, obs, deterministic=True):

        # Find the action matching the current CA
        idx = (
            np.abs(
                self.env.envs[0].history.ca
                - obs.flatten()[self.env.envs[0].observables.index("ca")]
                * self.env.envs[0].observable_scales["ca"]
            )
        ).idxmin()

        return [self.actions.loc[idx + 1].values], {}

    def save(self, name):
        # do nothing
        return 0

    def load(self, name):
        self.learn()
        return 0

    def generate_expert_traj(self, fname):
        df, total_reward = utilities.evaluate_agent(self.env, self)
        episode_starts = [False for _ in range(len(df))]
        episode_starts[-1] = True

        numpy_dict = {
            "actions": df[self.env.envs[0].action.actions].values,
            "obs": df[self.env.get_attr("observables", indices=0)[0]].values,
            "rewards": df.rewards.values,
            "episode_returns": np.array([total_reward]),
            "episode_starts": episode_starts,
        }
        np.savez(fname, **numpy_dict)

        return numpy_dict
