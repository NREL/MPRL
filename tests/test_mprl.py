# ========================================================================
#
# Imports
#
# ========================================================================
import os
import unittest
import numpy as np
import pandas as pd
import numpy.testing as npt
from stable_baselines.common.vec_env import DummyVecEnv
import mprl.engines as engines
import mprl.agents as agents
import mprl.utilities as utilities


# ========================================================================
#
# Test definitions
#
# ========================================================================
class MPRLTestCase(unittest.TestCase):
    """Tests for the engine environment and associated agents."""

    def setUp(self):

        self.T0, self.p0 = engines.calibrated_engine_ic()
        self.agentdir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "trained_agents"
        )

    def test_calibrated_agent(self):
        """Does the calibrated agent work as expected?"""

        # Initialize engine
        eng = engines.ContinuousTwoZoneEngine(
            T0=self.T0, p0=self.p0, nsteps=100, use_qdot=True
        )

        # Initialize the agent
        env = DummyVecEnv([lambda: eng])
        agent = agents.CalibratedAgent(env)
        agent.learn()

        # Evaluate the agent
        df, total_reward = utilities.evaluate_agent(env, agent)

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.00219521215114374)
        npt.assert_allclose(np.linalg.norm(df.p), 224.8448162626442)
        npt.assert_allclose(np.linalg.norm(df.T), 101005.70274507966)
        npt.assert_allclose(np.linalg.norm(df.rewards), 5.269396578180232)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.6641914874662471)

    def test_discrete_twozone_engine(self):
        """Does the DiscreteTwoZoneEngine work as expected?"""

        # Initialize engine
        eng = engines.DiscreteTwoZoneEngine(T0=self.T0, p0=self.p0, nsteps=21)

        # Evaluate a dummy agent that injects at a fixed time
        env = DummyVecEnv([lambda: eng])
        df = pd.DataFrame(
            0.0,
            index=eng.history.index,
            columns=list(
                dict.fromkeys(
                    eng.observables
                    + eng.internals
                    + eng.action.actions
                    + eng.histories
                    + ["rewards"]
                )
            ),
        )
        df[eng.histories] = eng.history[eng.histories]
        df.loc[0, ["rewards"]] = [engines.get_reward(eng.current_state)]

        # Evaluate actions from the agent in the environment
        obs = env.reset()
        df.loc[0, eng.observables] = obs
        df.loc[0, eng.internals] = eng.current_state[eng.internals]
        for index in eng.history.index[1:]:

            # Agent tries to inject twice, but is not allowed
            if (eng.current_state.ca == -10) or eng.current_state.ca == 10:
                action = [1]
            else:
                action = [0]
            obs, reward, done, info = env.step(action)

            # save history
            df.loc[index, eng.action.actions] = eng.action.current
            df.loc[index, eng.internals] = info[0]["internals"]
            df.loc[index, ["rewards"]] = reward
            df.loc[index, eng.observables] = obs
            if done:
                df.loc[index, eng.observables] = info[0]["terminal_observation"]
                break

        df = df.loc[:index, :]

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.0010489973276327207)
        npt.assert_allclose(np.linalg.norm(df.p), 115.96622593656717)
        npt.assert_allclose(np.linalg.norm(df.T), 12055.299832640749)
        npt.assert_allclose(np.linalg.norm(df.rewards), 2.0522427523102706)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.3)

    def test_reactor_engine(self):
        """Does the ReactorEngine work as expected?"""

        # Initialize engine
        eng = engines.ReactorEngine(T0=self.T0, p0=self.p0, dt=9e-6)

        # Evaluate a dummy agent that injects at a fixed time
        env = DummyVecEnv([lambda: eng])
        df = pd.DataFrame(
            0.0,
            index=eng.history.index,
            columns=list(
                dict.fromkeys(
                    eng.observables
                    + eng.internals
                    + eng.action.actions
                    + eng.histories
                    + ["rewards"]
                )
            ),
        )
        df[eng.histories] = eng.history[eng.histories]
        df.loc[0, ["rewards"]] = [engines.get_reward(eng.current_state)]

        # Evaluate actions from the agent in the environment
        obs = env.reset()
        df.loc[0, eng.observables] = obs
        df.loc[0, eng.internals] = eng.current_state[eng.internals]

        idx_inj = (eng.history.ca + 10).abs().idxmin()
        for index in eng.history.index[1:]:

            action = [1] if index == idx_inj else [0.0]
            obs, reward, done, info = env.step(action)

            # save history
            df.loc[index, eng.action.actions] = eng.action.current
            df.loc[index, eng.internals] = info[0]["internals"]
            df.loc[index, ["rewards"]] = reward
            df.loc[index, eng.observables] = obs
            if done:
                df.loc[index, eng.observables] = info[0]["terminal_observation"]
                break

        df = df.loc[:index, :]

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.010811037851028842)
        npt.assert_allclose(np.linalg.norm(df.p), 1279.6542882315912)
        npt.assert_allclose(np.linalg.norm(df.T), 62700.27309082788)
        npt.assert_allclose(np.linalg.norm(df.rewards), 25.366506367698257)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.3)


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    unittest.main()
