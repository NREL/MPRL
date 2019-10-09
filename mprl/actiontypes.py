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
    def __init__(self):
        super().__init__()
        self.actions = []
        self.size = len(self.actions)
        self.space = None
        self.masked = False
        self.scales = None
        self.current = None
        self.counter = None
        self.limit = None
        self.use_qdot = False
        self.max_mdot = 0.5
        self.max_qdot = np.finfo(np.float32).max

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
    def __init__(self, use_qdot):
        super(ContinuousActionType, self).__init__()
        self.use_qdot = use_qdot

        self.actions = ["mdot"]
        if self.use_qdot:
            self.actions.append("qdot")
        self.size = len(self.actions)

        if self.use_qdot:
            actions_low = np.array([0, -self.max_qdot])
            actions_high = np.array([self.max_mdot, self.max_qdot])
        else:
            actions_low = np.array([0])
            actions_high = np.array([self.max_mdot])
        self.space = spaces.Box(low=actions_low, high=actions_high, dtype=np.float16)

    def preprocess(self, action):
        """Preprocess the actions for use by engine"""
        action = self.parse(action)
        self.current = action
        return action


# ========================================================================
class DiscreteActionType(ActionType):
    def __init__(self, max_injections):
        super(DiscreteActionType, self).__init__()

        self.actions = ["mdot"]
        self.size = len(self.actions)
        self.scales = {"mdot": 0.3}
        self.counter = {"mdot": 0}
        self.limit = {"mdot": max_injections}
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
