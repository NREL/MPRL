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
        npt.assert_allclose(np.linalg.norm(df["T"]), 10389.766862402124)
        npt.assert_allclose(np.linalg.norm(df.rewards), 5.269396578180232)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.6641914874662471)
        npt.assert_allclose(np.linalg.norm(df.qdot), 97686.9157424243)

    def test_discrete_twozone_engine(self):
        """Does the DiscreteTwoZoneEngine work as expected?"""

        # Initialize engine
        eng = engines.DiscreteTwoZoneEngine(T0=self.T0, p0=self.p0, nsteps=201)
        env = DummyVecEnv([lambda: eng])
        variables = eng.observables + eng.internals + eng.histories
        df = pd.DataFrame(
            columns=list(dict.fromkeys(variables + eng.action.actions + ["rewards"]))
        )

        # Evaluate a dummy agent that injects at a fixed time
        done = False
        cnt = 0
        obs = env.reset()
        df.loc[cnt, variables] = eng.current_state[variables]
        df.loc[cnt, eng.action.actions] = 0
        df.loc[cnt, ["rewards"]] = [engines.get_reward(eng.current_state)]

        while not done:
            cnt += 1
            # Agent tries to inject twice, but is not allowed the second time
            action = (
                [1]
                if (eng.current_state.ca == -10) or eng.current_state.ca == 10
                else [0]
            )
            obs, reward, done, info = env.step(action)
            df.loc[cnt, variables] = info[0]["current_state"][variables]
            df.loc[cnt, eng.action.actions] = eng.action.current
            df.loc[cnt, ["rewards"]] = reward

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.003094822855555559)
        npt.assert_allclose(np.linalg.norm(df.p), 333.87423985351336)
        npt.assert_allclose(np.linalg.norm(df["T"]), 11588.56110434575)
        npt.assert_allclose(np.linalg.norm(df.rewards), 5.979798997051359)
        npt.assert_allclose(np.linalg.norm(df.mdot), 1.8)

    def test_reactor_engine(self):
        """Does the ReactorEngine work as expected?"""

        # Initialize engine
        eng = engines.ReactorEngine(
            T0=self.T0,
            p0=self.p0,
            agent_steps=201,
            dt=5e-6,
            rxnmech="dodecane_lu_nox.cti",
        )
        env = DummyVecEnv([lambda: eng])
        variables = eng.observables + eng.internals + eng.histories
        df = pd.DataFrame(
            columns=list(dict.fromkeys(variables + eng.action.actions + ["rewards"]))
        )

        # Evaluate a dummy agent that injects at a fixed time
        done = False
        cnt = 0
        obs = env.reset()
        df.loc[cnt, variables] = eng.current_state[variables]
        df.loc[cnt, eng.action.actions] = 0
        df.loc[cnt, ["rewards"]] = [engines.get_reward(eng.current_state)]

        while not done:
            cnt += 1
            # Agent tries to inject twice, but is not allowed the second time
            action = (
                [1]
                if (eng.current_state.ca == -10) or eng.current_state.ca == 10
                else [0]
            )
            obs, reward, done, info = env.step(action)
            df.loc[cnt, variables] = info[0]["current_state"][variables]
            df.loc[cnt, eng.action.actions] = eng.action.current
            df.loc[cnt, ["rewards"]] = reward

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.003094822855555559)
        npt.assert_allclose(np.linalg.norm(df.p), 352.60457507736834)
        npt.assert_allclose(np.linalg.norm(df["T"]), 17713.566835234808)
        npt.assert_allclose(np.linalg.norm(df.rewards), 6.7807147069970926)
        npt.assert_allclose(np.linalg.norm(df.mdot), 1.8)


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    unittest.main()
