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
        EOC_reward=False,
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

        # Do not let user modify the form of the penalty reward
        if "penalty" in names:
            print("Warning: user may not explicitly set the penalty reward. Stripping")
            names = [n for n in names if n != "penalty"]
            weights = [w for n, w in zip(names, weights) if n != "penalty"]
            norms = [m for n, m in zip(names, norms) if n != "penalty"]

        # Sanity check the weights
        if np.fabs(sum(weights) - 1.0) > 1e-13:
            sys.exit(f"""Weights don't sum to 1 ({sum(weights)} != 1))""")

        self.names = names + ["penalty"]
        self.n = len(self.names)
        self.set_norms(norms + [None])
        self.set_weights(weights + [1.0])
        self.set_weight_observables()
        self.negative_reward = negative_reward
        self.EOC_reward = EOC_reward
        self.randomize = randomize
        self.total_reward = {name: 0.0 for name in self.names}
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
                "normalized": lambda state, *unused: state["p"]
                * state["dV"]
                / self.norms["work"],
                "unnormalized": lambda state, *unused: state["p"] * state["dV"],
            },
            "nox": {
                "normalized": lambda state, *unused: -state["nox"]
                * state["dca"]
                / self.norms["nox"],
                "unnormalized": lambda state, *unused: -state["nox"] * state["dca"],
            },
            "soot": {
                "normalized": lambda state, *unused: -state["soot"]
                * state["dca"]
                / self.norms["soot"],
                "unnormalized": lambda state, *unused: -state["soot"] * state["dca"],
            },
            "penalty": {
                "unnormalized": lambda state, nsteps, penalty: (
                    self.negative_reward / (nsteps - 1) if penalty else 0.0
                )
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

    def summer(self, state, nsteps, penalty):
        """Summation of the rewards

        :param state: environment state
        :type state: dict
        :param nsteps: number of steps
        :type nsteps: int
        :returns: reward
        :rtype: float
        """
        return sum(self.compute(state, nsteps, penalty).values())

    def compute(self, state, nsteps, penalty, done):
        """Compute each weighted reward

        :param state: environment state
        :type state: dict
        :param nsteps: number of steps
        :type nsteps: int
        :param penalty: whether to penalize the reward
        :type penalty: bool
        :returns: reward
        :rtype: float
        """

        self.total_reward = {
            n: self.total_reward[n]
            + self.weights[n] * self.rewards[n](state, nsteps, penalty)
            for n in self.names
        }

        if self.EOC_reward:
            return {n: self.total_reward[n] if done else 0.0 for n in self.names}
        else:
            return {
                n: self.weights[n] * self.rewards[n](state, nsteps, penalty)
                if n == "work"
                else (self.total_reward[n] if done else 0.0)
                for n in self.names
            }

    def set_norms(self, norms):
        if len(norms) != self.n:
            sys.exit(f"""Norms length != names ({len(norms)} != {self.n})""")
        self.norms = {name: norm for name, norm in zip(self.names, norms)}

    def set_weights(self, weights):
        if len(weights) != self.n:
            sys.exit(f"""Weights length != names ({len(weights)} != {self.n})""")
        self.weights = {name: weight for name, weight in zip(self.names, weights)}

    def set_weight_observables(self):
        """Define which weights should be observables for the engine"""

        # If there is the penalty reward and another reward, don't
        # bother adding observables for the engine
        if self.n <= 2:
            self.is_observable = {k: False for k in self.weights.keys()}
        else:
            self.is_observable = {k: True for k in self.weights.keys()}
            self.is_observable["penalty"] = False

    def set_random_weights(self):
        """Set random weights

        Uniform sampling of the weights. This problem in general is
        called simplex sampling. This seems to cause a lot of
        confusion online. Basically you want to take a uniform sample
        from the set X = { (x1, x2, ..., xD) | 0 <= xi <= 1, x1 + x2 +
        ... + xD = 1}

        It comes down to a special case of sampling from a Dirichlet
        distribution. The concentrations are set to 1 to get a uniform
        sampling.

        This link has a good visual for what would happen if you
        didn't do things correctly:
        https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex

        Randomize all the weights except the one for the negative
        reward (explicitly set to 1)

        """
        self.set_weights(np.hstack((np.random.dirichlet(np.ones(self.n - 1)), [1.0])))

    def get_observable_attributes(self):
        """Return the weight observable attributes"""
        return {
            f"""w_{k}""": {"low": 0.0, "high": 1.0, "scale": 1.0}
            for k in self.weights.keys()
            if self.is_observable[k]
        }

    def get_observables(self):
        """Return the weight observables"""
        return [f"""w_{k}""" for k in self.weights.keys() if self.is_observable[k]]

    def get_rewards(self):
        """Return the formatted reward names"""
        return [f"""r_{k}""" for k in self.rewards.keys()]

    def get_state_updater(self):
        # Ideally we would do:
        # return {f"""w_{k}""": lambda: v for k, v in self.weights.items()}
        # but for some reason it doesn't evaluate properly
        return {
            "w_work": lambda: self.weights["work"],
            "w_nox": lambda: self.weights["nox"],
            "w_soot": lambda: self.weights["soot"],
            "w_penalty": lambda: self.weights["penalty"],
        }

    def get_state_reseter(self):
        return self.get_state_updater()

    def reset(self):
        """Reset the reward weights"""
        if self.randomize:
            self.set_random_weights()
        self.total_reward = {name: 0.0 for name in self.names}
