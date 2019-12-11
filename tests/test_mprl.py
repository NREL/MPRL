# ========================================================================
#
# Imports
#
# ========================================================================
import os
import sys
import time
import unittest
import numpy as np
import pandas as pd
import numpy.testing as npt
from stable_baselines.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
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

        self.agentdir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "trained_agents"
        )

    def tearDown(self):
        utilities.save_plots("tests.pdf")

    def test_calibrated_agent(self):
        """Does the calibrated agent work as expected?"""

        # Initialize engine
        eng = engines.ContinuousTwoZoneEngine(
            agent_steps=100,
            use_qdot=True,
            fuel="PRF100",
            rxnmech="llnl_gasoline_surrogate_323.xml",
        )

        # Initialize the agent
        env = DummyVecEnv([lambda: eng])
        agent = agents.CalibratedAgent(env)
        agent.learn()

        # Evaluate the agent
        t0 = time.time()
        df, total_reward = utilities.evaluate_agent(env, agent)
        elapsed = time.time() - t0
        utilities.plot_df(env, df, idx=0, name="calibrated")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.0021952121511437405)
        npt.assert_allclose(np.linalg.norm(df.p), 22012100.1714362)
        npt.assert_allclose(np.linalg.norm(df["T"]), 14210.479334973)
        npt.assert_allclose(np.linalg.norm(df.rewards), 104.47361753965)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.04144043816723)
        npt.assert_allclose(np.linalg.norm(df.qdot), 97686.915742424)
        print(f"Wall time for CalibratedAgent = {elapsed} seconds")

    def test_exhaustive_agent(self):
        """Does the exhaustive agent work as expected?"""

        # Initialize engine
        eng = engines.DiscreteTwoZoneEngine(
            agent_steps=101,
            mdot=0.1,
            max_minj=5e-5,
            fuel="dodecane",
            rxnmech="dodecane_lu_nox.cti",
            negative_reward=-0.05,
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
        t0 = time.time()
        df, total_reward = utilities.evaluate_agent(env, agent)
        elapsed = time.time() - t0
        utilities.plot_df(env, df, idx=1, name="exhaustive")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.002205916821815495)
        npt.assert_allclose(np.linalg.norm(df.p), 37808279.50349433)
        npt.assert_allclose(np.linalg.norm(df["T"]), 21061.100819984735)
        npt.assert_allclose(np.linalg.norm(df.rewards), 153.19968366157858)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.14142135623730953)
        print(f"Wall time for ExhaustiveAgent = {elapsed} seconds")

    def test_discrete_twozone_engine(self):
        """Does the DiscreteTwoZoneEngine work as expected?"""

        # Initialize engine
        eng = engines.DiscreteTwoZoneEngine(
            agent_steps=101,
            fuel="PRF100",
            rxnmech="llnl_gasoline_surrogate_323.xml",
            mdot=0.1,
            max_minj=5e-5,
            negative_reward=-0.05,
        )
        env = DummyVecEnv([lambda: eng])
        variables = eng.observables + eng.internals + eng.histories
        df = pd.DataFrame(
            columns=list(dict.fromkeys(variables + eng.action.actions + ["rewards"]))
        )

        # Evaluate a dummy agent that injects at a fixed time
        t0 = time.time()
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
                if (eng.current_state.ca == -2) or eng.current_state.ca == 2
                else [0]
            )
            obs, reward, done, info = env.step(action)
            df.loc[cnt, variables] = info[0]["current_state"][variables]
            df.loc[cnt, eng.action.actions] = eng.action.current
            df.loc[cnt, ["rewards"]] = reward

        elapsed = time.time() - t0

        utilities.plot_df(env, df, idx=2, name="discrete")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.002205916821815495)
        npt.assert_allclose(np.linalg.norm(df.p), 37091026.33424249)
        npt.assert_allclose(np.linalg.norm(df["T"]), 21272.660424423644)
        npt.assert_allclose(np.linalg.norm(df.rewards), 155.27880219948767)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.14142135623730953)
        print(f"Wall time for DiscreteTwoZoneEngine = {elapsed} seconds")

    def test_discrete_twozone_engine_with_delay(self):
        """Does the DiscreteTwoZoneEngine with injection delay work as expected?"""

        # Initialize engine
        eng = engines.DiscreteTwoZoneEngine(
            agent_steps=101,
            fuel="PRF100",
            rxnmech="llnl_gasoline_surrogate_323.xml",
            mdot=0.1,
            max_minj=5e-5,
            injection_delay=0.0025,
        )
        env = DummyVecEnv([lambda: eng])
        variables = eng.observables + eng.internals + eng.histories
        df = pd.DataFrame(
            columns=list(dict.fromkeys(variables + eng.action.actions + ["rewards"]))
        )

        # Evaluate a dummy agent that injects at a fixed time
        t0 = time.time()
        done = False
        cnt = 0
        obs = env.reset()
        df.loc[cnt, variables] = eng.current_state[variables]
        df.loc[cnt, eng.action.actions] = 0
        df.loc[cnt, ["rewards"]] = [engines.get_reward(eng.current_state)]

        while not done:
            cnt += 1
            # Agent tries to inject thrice, but is not allowed the second time
            action = (
                [1]
                if (eng.current_state.ca == -10)
                or eng.current_state.ca == 10
                or eng.current_state.ca == 15
                else [0]
            )
            obs, reward, done, info = env.step(action)
            df.loc[cnt, variables] = info[0]["current_state"][variables]
            df.loc[cnt, eng.action.actions] = eng.action.current
            df.loc[cnt, ["rewards"]] = reward

        elapsed = time.time() - t0

        utilities.plot_df(env, df, idx=5, name="DiscreteTwoZone (delay)")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.002205916821815495)
        npt.assert_allclose(np.linalg.norm(df.p), 29368413.21384358)
        npt.assert_allclose(np.linalg.norm(df["T"]), 14499.415561284017)
        npt.assert_allclose(np.linalg.norm(df.rewards), 102.80455421172557)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.1)
        print(f"Wall time for DiscreteTwoZoneEngine with delay = {elapsed} seconds")

    def test_reactor_engine(self):
        """Does the ReactorEngine work as expected?"""

        # Initialize engine
        eng = engines.ReactorEngine(
            agent_steps=101,
            Tinj=300.0,
            dt=4e-6,
            rxnmech="dodecane_lu_nox.cti",
            mdot=0.1,
            max_minj=5e-5,
            negative_reward=-0.05,
        )
        env = DummyVecEnv([lambda: eng])
        variables = eng.observables + eng.internals + eng.histories
        df = pd.DataFrame(
            columns=list(dict.fromkeys(variables + eng.action.actions + ["rewards"]))
        )

        # Evaluate a dummy agent that injects at a fixed time
        t0 = time.time()
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

        elapsed = time.time() - t0

        utilities.plot_df(env, df, idx=3, name="reactor")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.002205916821815495)
        npt.assert_allclose(np.linalg.norm(df.p), 39472544.14618649, rtol=1e-5)
        npt.assert_allclose(np.linalg.norm(df["T"]), 13040.23077865307, rtol=1e-5)
        npt.assert_allclose(np.linalg.norm(df.rewards), 148.66188240588772, rtol=1e-5)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.14142135623730953)
        print(f"Wall time for ReactorEngine = {elapsed} seconds")

    def test_equilibrate_engine(self):
        """Does the EquilibrateEngine work as expected?"""

        # Initialize engine
        eng = engines.EquilibrateEngine(
            agent_steps=101,
            Tinj=300.0,
            rxnmech="dodecane_lu_nox.cti",
            mdot=0.1,
            max_minj=5e-5,
            negative_reward=-0.05,
        )
        env = DummyVecEnv([lambda: eng])
        variables = eng.observables + eng.internals + eng.histories
        df = pd.DataFrame(
            columns=list(dict.fromkeys(variables + eng.action.actions + ["rewards"]))
        )

        # Evaluate a dummy agent that injects at a fixed time
        t0 = time.time()
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

        elapsed = time.time() - t0

        utilities.plot_df(env, df, idx=4, name="EQ")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.002205916821815495)
        npt.assert_allclose(np.linalg.norm(df.p), 44024925.44422519)
        npt.assert_allclose(np.linalg.norm(df["T"]), 12768.241249831073)
        npt.assert_allclose(np.linalg.norm(df.rewards), 159.80034835184017)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.14142135623730953)
        print(f"Wall time for EquilibrateEngine = {elapsed} seconds")


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    unittest.main()
