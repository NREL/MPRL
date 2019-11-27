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
        self.current = None
        self.masked = False
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
    def __init__(self, action, scales, limits, delays):
        super(DiscreteActionType, self).__init__(action)

        self.scales = scales
        self.limits = limits
        self.delays = delays
        self.attempt_counter = None
        self.success_counter = None
        self.success_time = None  # time since last successful action
        self.space = spaces.Discrete(2)
        self.reset()

    def preprocess(self, action):
        """Preprocess the actions for use by engine"""
        action = self.parse(action)
        self.count_attempt(action)
        action = self.scale(action)
        action = self.mask(action)
        self.current = action
        self.count_success(action)

        return action

    def scale(self, action):
        """Scale discrete actions to physical space"""
        for key in action:
            action[key] *= self.scales[key]
        return action

    def count_attempt(self, action):
        """Keep a running counter of attempted discrete actions"""
        for key in action:
            self.attempt_counter[key] += action[key]

    def count_success(self, action):
        """Keep a running counter of successful discrete actions and increment
           time since last successful action"""
        for key in self.actions:
            if action[key] > 0:
                self.success_counter[key] += 1
                self.success_time[key] = 0
            else:
                self.success_time[key] += 1

    def mask(self, action):
        """Mask the action if necessary"""
        self.masked = False
        allowed = self.isallowed()
        for key in action:
            if not allowed[key] and action[key] > 0:
                self.masked = True
                action[key] = 0.0

        return action

    def reset(self):
        """Reset the actions"""
        self.masked = False
        self.attempt_counter = {key: 0 for key in self.actions}
        self.success_counter = {key: 0 for key in self.actions}
        self.success_time = {key: np.iinfo(np.int32).max // 2 for key in self.actions}

    def isallowed(self):
        """Check if the action is allowed"""
        allowed = {key: True for key in self.actions}
        for key in allowed:
            if (self.success_counter[key] >= self.limits[key]) or (
                self.success_time[key] <= self.delays[key]
            ):
                allowed[key] = False
        return allowed
