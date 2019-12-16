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
# Functions
#
# ========================================================================
def print_norms(df, precision=8):
    print("Printing the norms of each column in the dataframe")
    for col in df.columns:
        print(f"""{col}: {np.linalg.norm(df[col]):.{precision}f}""")


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
        npt.assert_allclose(np.linalg.norm(df.V), 0.002195212151)
        npt.assert_allclose(np.linalg.norm(df.p), 21773583.08233403)
        npt.assert_allclose(np.linalg.norm(df["T"]), 13977.16741776)
        npt.assert_allclose(np.linalg.norm(df.rewards), 103.20006525)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.04144044)
        npt.assert_allclose(np.linalg.norm(df.qdot), 97686.91574242)
        print(f"Wall time for CalibratedAgent = {elapsed} seconds")

    def test_exhaustive_agent(self):
        """Does the exhaustive agent work as expected?"""

        # Initialize engine
        eng = engines.DiscreteTwoZoneEngine(
            agent_steps=101,
            mdot=0.1,
            max_minj=2.5e-5,
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
        npt.assert_allclose(np.linalg.norm(df.p), 25431213.9403193)
        npt.assert_allclose(np.linalg.norm(df["T"]), 13611.586979159425)
        npt.assert_allclose(np.linalg.norm(df.rewards), 101.4137388027926)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.1)
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
            negative_reward=-101,
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
            # Agent injects twice
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

        utilities.plot_df(env, df, idx=2, name="discrete")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.002205916821815495)
        npt.assert_allclose(np.linalg.norm(df.p), 36686979.19585361)
        npt.assert_allclose(np.linalg.norm(df["T"]), 20924.35004286)
        npt.assert_allclose(np.linalg.norm(df.rewards), 153.41477580)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.14142136)
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
            negative_reward=-101,
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
                or eng.current_state.ca == 16
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
        npt.assert_allclose(np.linalg.norm(df.p), 34770281.93257003)
        npt.assert_allclose(np.linalg.norm(df["T"]), 20627.76382495)
        npt.assert_allclose(np.linalg.norm(df.rewards), 151.28886436)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.14142135623730953)
        print(f"Wall time for DiscreteTwoZoneEngine with delay = {elapsed} seconds")

    def test_reactor_engine(self):
        """Does the ReactorEngine work as expected?"""

        # Initialize engine
        eng = engines.ReactorEngine(
            agent_steps=101,
            Tinj=300.0,
            target_dt=4e-6,
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
