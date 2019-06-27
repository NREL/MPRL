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

        return [self.actions.loc[idx + 1, ["mdot", "qdot"]].values]
