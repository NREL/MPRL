# ========================================================================
#
# Imports
#
# ========================================================================
import sys
import numpy as np
from abc import ABC, abstractmethod
from gym import spaces


# ========================================================================
#
# Classes
#
# ========================================================================
class ActionType(ABC):
    def __init__(self, actions):
        super().__init__()
        self.actions = actions
        self.size = len(self.actions)
        self.space = None
        self.masked = False
        self.scales = None
        self.current = None
        self.counter = None
        self.limit = None
        self.use_qdot = False

    def parse(self, action):
        """Create dictionary of actions"""
        action = np.array(action).flatten()
        if len(action) != self.size:
            sys.exit(f"Error: invalid action size {len(action)} != {self.size}")

        dct = {}
        for k, name in enumerate(self.actions):
            dct[name] = action[k]
        return dct

    def symmetrize_space(self):
        """Make action space symmetric (e.g. for DDPG)"""
        self.space.low = -self.space.high

    def reset(self):
        pass

    @abstractmethod
    def preprocess(self, action):
        pass


# ========================================================================
class ContinuousActionType(ActionType):
    def __init__(self, actions):
        super(ContinuousActionType, self).__init__(actions)

        self.use_qdot = "qdot" in actions

        mins = {"mdot": 0.0, "qdot": -np.finfo(np.float32).max}
        maxs = {"mdot": 0.5, "qdot": np.finfo(np.float32).max}

        actions_low = np.array([mins[key] for key in actions])
        actions_high = np.array([maxs[key] for key in actions])
        self.space = spaces.Box(low=actions_low, high=actions_high, dtype=np.float16)

    def preprocess(self, action):
        """Preprocess the actions for use by engine"""
        action = self.parse(action)
        self.current = action
        return action


# ========================================================================
class DiscreteActionType(ActionType):
    def __init__(self, actions, scales, limits):
        super(DiscreteActionType, self).__init__(actions)

        self.scales = scales
        self.limit = limits
        self.counter = {key: 0 for key in self.actions}
        self.space = spaces.Discrete(2)

    def preprocess(self, action):
        """Preprocess the actions for use by engine"""
        action = self.parse(action)
        self.count(action)
        action = self.scale(action)
        action = self.mask(action)
        self.current = action

        return action

    def scale(self, action):
        """Scale discrete actions to physical space"""
        for key in action:
            action[key] *= self.scales[key]
        return action

    def count(self, action):
        """Keep a running counter of discrete actions"""
        for key in action:
            self.counter[key] += action[key]

    def mask(self, action):
        """Mask the action if necessary"""
        self.masked = False
        for key in action:
            if self.counter[key] > self.limit[key] and action[key] > 0:
                print("Injection not allowed!")
                self.masked = True
                action[key] = 0.0

        return action

    def reset(self):
        """Reset the actions"""
        self.masked = False
        for key in self.actions:
            self.counter[key] = 0
