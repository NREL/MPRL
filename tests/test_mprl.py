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

    def tearDown(self):
        utilities.save_plots("tests.pdf")

    def test_calibrated_agent(self):
        """Does the calibrated agent work as expected?"""

        # Initialize engine
        eng = engines.ContinuousTwoZoneEngine(
            T0=self.T0,
            p0=self.p0,
            nsteps=100,
            use_qdot=True,
            fuel="PRF100",
            rxnmech="llnl_gasoline_surrogate_323.xml",
        )

        # Initialize the agent
        env = DummyVecEnv([lambda: eng])
        agent = agents.CalibratedAgent(env)
        agent.learn()

        # Evaluate the agent
        df, total_reward = utilities.evaluate_agent(env, agent)
        utilities.plot_df(env, df, idx=0, name="calibrated")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.0021952121511437405)
        npt.assert_allclose(np.linalg.norm(df.p), 23562451.47495935)
        npt.assert_allclose(np.linalg.norm(df["T"]), 9546.3503213235)
        npt.assert_allclose(np.linalg.norm(df.rewards), 109.90666061638017)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.6641914874662471)
        npt.assert_allclose(np.linalg.norm(df.qdot), 97686.9157424243)

    def test_exhaustive_agent(self):
        """Does the exhaustive agent work as expected?"""

        # Initialize engine
        eng = engines.DiscreteTwoZoneEngine(
            T0=self.T0,
            p0=self.p0,
            nsteps=101,
            fuel="dodecane",
            rxnmech="dodecane_lu_nox.cti",
            small_negative_reward=-0.05,
        )
        env = DummyVecEnv([lambda: eng])
        variables = eng.observables + eng.internals + eng.histories
        df = pd.DataFrame(
            columns=list(dict.fromkeys(variables + eng.action.actions + ["rewards"]))
        )

        # Initialize the agent
        agent = agents.ExhaustiveAgent(env)
        agent.learn()

        # Evaluate the agent
        df, total_reward = utilities.evaluate_agent(env, agent)
        utilities.plot_df(env, df, idx=1, name="exhaustive")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.002205916821815495)
        npt.assert_allclose(np.linalg.norm(df.p), 21672145.315614562)
        npt.assert_allclose(np.linalg.norm(df["T"]), 7245.285965190272)
        npt.assert_allclose(np.linalg.norm(df.rewards), 83.4323700059674)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.9)

    def test_discrete_twozone_engine(self):
        """Does the DiscreteTwoZoneEngine work as expected?"""

        # Initialize engine
        eng = engines.DiscreteTwoZoneEngine(
            T0=self.T0,
            p0=self.p0,
            nsteps=201,
            fuel="PRF100",
            rxnmech="llnl_gasoline_surrogate_323.xml",
            small_negative_reward=-0.05,
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

        utilities.plot_df(env, df, idx=2, name="ppo")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.003094822855555559)
        npt.assert_allclose(np.linalg.norm(df.p), 34353773.82674564)
        npt.assert_allclose(np.linalg.norm(df["T"]), 11479.49800333749)
        npt.assert_allclose(np.linalg.norm(df.rewards), 61.69149868914742)
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
            small_negative_reward=-0.05,
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

        utilities.plot_df(env, df, idx=3, name="ppo")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.003094822855555559)
        npt.assert_allclose(np.linalg.norm(df.p), 34947820.529421)
        npt.assert_allclose(np.linalg.norm(df["T"]), 17690.180765)
        npt.assert_allclose(np.linalg.norm(df.rewards), 70.275994)
        npt.assert_allclose(np.linalg.norm(df.mdot), 1.8)


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    unittest.main()
