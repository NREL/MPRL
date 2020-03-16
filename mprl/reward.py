# ========================================================================
#
# Imports
#
# ========================================================================
import sys
import copy
import numpy as np


# ========================================================================
#
# Classes
#
# ========================================================================
class Reward:
    def __init__(
        self,
        names=["work"],
        norms=[None],
        weights=[1.0],
        negative_reward=-800.0,
        randomize=False,
    ):
        """Initialize Reward

        :param names: name of reward types
        :type names: list
        :param norms: values to normalize the reward
        :type norms: list
        :param weights: reward weights
        :type weights: list
        :param negative_reward: negative reward for illegal actions
        :type negative_reward: float
        """

        self.names = names
        self.n = len(self.names)
        self.set_norms(norms)
        self.set_weights(weights)
        self.negative_reward = negative_reward
        self.randomize = randomize
        self.setup_reward()

    def __repr__(self):
        return self.describe()

    def __str__(self):
        return f"""An instance of {self.describe()}"""

    def __deepcopy__(self, memo):
        """Deepcopy implementation"""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            try:
                setattr(result, k, copy.deepcopy(v, memo))
            except (NotImplementedError, TypeError):
                sys.exit(f"ERROR: in deepcopy of {self.__class__.__name__}")
        result.setup_reward()
        return result

    def __getstate__(self):
        """Copy the object's state from self.__dict__"""
        state = self.__dict__.copy()

        # Remove the unpicklable entries.
        for v in self.lambda_names:
            del state[v]

        return state

    def __setstate__(self, state):
        """Restore instance attributes"""
        self.__dict__.update(state)

        # Repopulate the unpicklable entries
        self.setup_reward()

    def describe(self):
        return f"""{self.__class__.__name__}(names={self.names}, norms={list(self.norms.values())}, weights={list(self.weights.values())}, negative_reward={self.negative_reward}, randomize={self.randomize})"""

    def setup_reward(self):
        """Setup reward function

        The reason for doing it like this is to enable object pickling
        with the standard pickle library. Basically we skip pickling
        these and then manually add them back in.

        """
        self.lambda_names = ["available_rewards", "rewards"]
        self.available_rewards = {
            "work": {
                "normalized": lambda state, nsteps: (
                    state["p"] * state["dV"] - self.norms["work"] / (nsteps - 1)
                )
                / self.norms["work"],
                "unnormalized": lambda state, nsteps: state["p"] * state["dV"],
            },
            "nox": {
                "normalized": lambda state, nsteps: (
                    self.norms["nox"] / (nsteps - 1) - state["nox"]
                )
                / self.norms["nox"],
                "unnormalized": lambda state, nsteps: -state["nox"],
            },
            "soot": {
                "normalized": lambda state, nsteps: (
                    self.norms["soot"] / (nsteps - 1) - state["soot"]
                )
                / self.norms["soot"],
                "unnormalized": lambda state, nsteps: -state["soot"],
            },
        }

        self.rewards = {}
        for name in self.names:
            if name not in self.available_rewards.keys():
                sys.exit(f"""Non existing reward type: {name}""")
            self.rewards[name] = (
                self.available_rewards[name]["normalized"]
                if self.norms[name] is not None
                else self.available_rewards[name]["unnormalized"]
            )

    def evaluate(self, state, nsteps):
        """Evaluate the reward

        :param state: environment state
        :type state: dict
        :param nsteps: number of steps
        :type nsteps: int
        :returns: reward
        :rtype: float
        """
        return sum(self.compute(state, nsteps))

    def compute(self, state, nsteps):
        """Compute each weighted reward

        :param state: environment state
        :type state: dict
        :param nsteps: number of steps
        :type nsteps: int
        :returns: reward
        :rtype: float
        """
        return [self.weights[n] * self.rewards[n](state, nsteps) for n in self.names]

    def set_norms(self, norms):
        if len(norms) != self.n:
            sys.exit(f"""Norms length != names ({len(norms)} != {self.n})""")
        self.norms = {name: norm for name, norm in zip(self.names, norms)}

    def set_weights(self, weights):
        if len(weights) != self.n:
            sys.exit(f"""Weights length != names ({len(weights)} != {self.n})""")
        if np.fabs(sum(weights) - 1.0) > 1e-13:
            sys.exit(f"""Weights don't sum to 1 ({sum(weights)} != 1))""")
        self.weights = {name: weight for name, weight in zip(self.names, weights)}

    def set_random_weigths(self, precision=1000):
        """Set random weights

        Make a random vector from a uniform distribution where is sums
        to 1. This is a tricky problem in general (drawing from a
        uniform and then normalizing won't lead to a uniform
        distribution of vectors). This trick might actually do it
        according to the web. But it works for ints, so I adapted it
        to draw ints to sum to a number (precision) and then normalize
        by that number. That should give me a uniformly drawn vector
        in [0,1] that sums to 1 (with the caveat that it is "binned"
        by the precision).
        """
        self.set_weights(
            np.random.multinomial(
                precision, [1 / np.float(self.n)] * self.n, size=1
            ).flatten()
            / precision
        )

    def get_observable_attributes(self):
        """Return the weight observable attributes"""
        return {
            f"""w_{k}""": {"low": 0.0, "high": 1.0, "scale": 1.0} for k in self.names
        }

    def get_observables(self):
        """Return the weight observables (only if more than one)"""
        if self.n > 1:
            return [f"""w_{k}""" for k in self.names]
        else:
            return []

    def get_rewards(self):
        """Return the formatted reward names (only if more than one)"""
        if self.n > 1:
            return [f"""r_{k}""" for k in self.names]
        else:
            return []

    def get_state_updater(self):
        # Ideally we would do:
        # return {f"""w_{k}""": lambda: v for k, v in self.weights.items()}
        # but for some reason it doesn't evaluate properly
        return {
            "w_work": lambda: self.weights["work"],
            "w_nox": lambda: self.weights["nox"],
            "w_soot": lambda: self.weights["soot"],
        }

    def get_state_reseter(self):
        return self.get_state_updater()

    def reset(self):
        """Reset the reward weights"""
        if self.randomize:
            self.set_random_weigths()
