# ========================================================================
#
# Imports
#
# ========================================================================
import os
import sys
import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
from multiprocessing import Pool
import pickle
from abc import ABC, abstractmethod
from stable_baselines.common.vec_env import DummyVecEnv
import mprl.utilities as utilities


# ========================================================================
#
# Classes
#
# ========================================================================
class Agent(ABC):
    def __init__(self, env):
        # Only support DummyVecEnv
        if not isinstance(env, DummyVecEnv):
            sys.exit("Please use DummyVecEnv for this agent")

        self.env = env
        self.eng = self.env.envs[0]

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def predict(self, obs, **kwargs):
        pass

    @abstractmethod
    def save(self, name):
        pass

    @abstractmethod
    def load(self, name, env):
        pass


# ========================================================================
class CalibratedAgent(Agent):
    def __init__(self, env):
        Agent.__init__(self, env)
        self.datadir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "datafiles"
        )

    def learn(self):
        self.actions = utilities.interpolate_df(
            self.eng.history["ca"],
            "ca",
            pd.read_csv(os.path.join(self.datadir, "calibrated_data.csv")),
        )
        self.actions.index = self.eng.history["index"]
        self.actions.mdot[self.actions.mdot < 0] = 0
        self.actions = self.actions[self.eng.action.actions]

    def predict(self, obs, **kwargs):

        # Find the action matching the current CA
        idx = np.argmin(
            np.abs(
                self.eng.history["ca"]
                - obs.flatten()[self.eng.observables.index("ca")]
                * self.eng.observable_attributes["ca"]["scale"]
            )
        )

        return [self.actions.loc[idx + 1].values], {}

    def save(self, name):
        # do nothing
        return 0

    def load(self, name, env):
        self.env = env
        self.learn()
        return 0

    def generate_expert_traj(self, fname):
        df, total_reward = utilities.evaluate_agent(self.env, self)
        episode_starts = [False for _ in range(len(df))]
        episode_starts[-1] = True

        numpy_dict = {
            "actions": df[self.eng.action.actions].values,
            "obs": df[self.env.get_attr("observables", indices=0)[0]].values,
            "rewards": df.rewards.values,
            "episode_returns": np.array([total_reward]),
            "episode_starts": episode_starts,
        }
        np.savez(fname, **numpy_dict)

        return numpy_dict


# ========================================================================
class ExhaustiveAgent(Agent):
    def __init__(self, env):
        Agent.__init__(self, env)
        self.datadir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "datafiles"
        )
        self.max_ninj = self.eng.max_injections
        self.best_inj = ()

    def learn(self, nranks=1):

        assert (
            nranks <= mp.cpu_count()
        ), f"Number of ranks ({nranks}) are greater than the number of available ranks ({mp.cpu_count()})"
        # Loop over all possible injection CAs
        agent_ca = self.eng.history["ca"]

        envlst = [self.env] * nranks
        chunk = (
            sum(1 for _ in itertools.combinations(agent_ca, self.max_ninj)) // nranks
        )
        injections = utilities.grouper(
            itertools.combinations(agent_ca, self.max_ninj), chunk
        )
        with Pool(processes=nranks) as pool:
            result = pool.starmap(self.evaluate_injections, zip(injections, envlst))

        # reduce the best injection and reward
        best_reward = -np.finfo(np.float32).max
        for d in result:
            if d["reward"] > best_reward:
                self.best_inj = d["inj"]
                best_reward = d["reward"]

    @staticmethod
    def evaluate_injections(injections, env):
        best_reward = -np.finfo(np.float32).max
        best_inj = ()
        eng = env.envs[0]
        for inj in injections:
            done = [False]
            obs = env.reset()
            total_reward = 0
            while not done[0]:
                action = [1] if (eng.current_state["ca"] in inj) else [0]
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]

            if total_reward > best_reward:
                best_reward = total_reward
                best_inj = inj
        return {"reward": best_reward, "inj": best_inj}

    def predict(self, obs, **kwargs):

        current_ca = obs[0][0] * self.eng.observable_attributes["ca"]["scale"]
        tol = 1e-4
        action = [0]
        for inj in self.best_inj:
            if np.fabs(inj - current_ca) < tol:
                action = [1]
                break

        return action, {}

    def save(self, name):
        with open(name + ".pkl", "wb") as f:
            pickle.dump(self.best_inj, f)
        return 0

    def load(self, name, env):
        self.env = env
        with open(name, "rb") as f:
            self.best_inj = pickle.load(f)
        return 0
