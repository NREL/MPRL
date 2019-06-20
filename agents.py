# ========================================================================
#
# Imports
#
# ========================================================================
import os
import random
from collections import deque
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import utilities


# ========================================================================
#
# Classes
#
# ========================================================================
class Agent:
    def __init__(self, env):
        self.state_size = len(env.observables)
        self.action_size = env.action_size
        self.env = env


# ========================================================================
class CalibratedAgent(Agent):
    def __init__(self, env):
        Agent.__init__(self, env)

    def train(self):
        self.actions = utilities.interpolate_df(
            self.env.states.ca,
            "ca",
            pd.read_csv(os.path.join("datafiles", "calibrated_data.csv")),
        )
        self.actions.index = self.env.states.index

    def act(self, state):
        return self.actions.loc[state.name + 1, ["mdot", "qdot"]]


# ========================================================================
class DQNAgent(Agent):
    def __init__(self, env):
        Agent.__init__(self, env)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_episodes = 1
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        # FIXME, qdot should be set to zero
        # FIXME, the high bound should be fixed
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(low=0.0, high=10, size=self.action_size)

        return self.model.predict(np.reshape(state.values, [1, -1])).flatten()

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
