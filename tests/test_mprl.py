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
        npt.assert_allclose(np.linalg.norm(df.p), 22135415.810781036)
        npt.assert_allclose(np.linalg.norm(df["T"]), 14294.35906971223)
        npt.assert_allclose(np.linalg.norm(df.rewards), 105.13630723422598)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.6641914874662471)
        npt.assert_allclose(np.linalg.norm(df.qdot), 97686.9157424243)
        print(f"Wall time for CalibratedAgent = {elapsed} seconds")

    def test_exhaustive_agent(self):
        """Does the exhaustive agent work as expected?"""

        # Initialize engine
        eng = engines.DiscreteTwoZoneEngine(
            T0=self.T0,
            p0=self.p0,
            agent_steps=101,
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
        t0 = time.time()
        df, total_reward = utilities.evaluate_agent(env, agent)
        elapsed = time.time() - t0
        utilities.plot_df(env, df, idx=1, name="exhaustive")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.002205916821815495)
        npt.assert_allclose(np.linalg.norm(df.p), 20166551.0606587)
        npt.assert_allclose(np.linalg.norm(df["T"]), 10702.697172931328)
        npt.assert_allclose(np.linalg.norm(df.rewards), 80.00996887426106)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.9)
        print(f"Wall time for ExhaustiveAgent = {elapsed} seconds")

    def test_discrete_twozone_engine(self):
        """Does the DiscreteTwoZoneEngine work as expected?"""

        # Initialize engine
        eng = engines.DiscreteTwoZoneEngine(
            T0=self.T0,
            p0=self.p0,
            agent_steps=201,
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

        utilities.plot_df(env, df, idx=2, name="discrete")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.003094822855555559)
        npt.assert_allclose(np.linalg.norm(df.p), 32287395.509347167)
        npt.assert_allclose(np.linalg.norm(df["T"]), 15696.023931640237)
        npt.assert_allclose(np.linalg.norm(df.rewards), 58.310239)
        npt.assert_allclose(np.linalg.norm(df.mdot), 1.8)
        print(f"Wall time for DiscreteTwoZoneEngine = {elapsed} seconds")

    def test_discrete_twozone_engine_with_delay(self):
        """Does the DiscreteTwoZoneEngine with injection delay work as expected?"""

        # Initialize engine
        eng = engines.DiscreteTwoZoneEngine(
            T0=self.T0,
            p0=self.p0,
            agent_steps=201,
            fuel="PRF100",
            rxnmech="llnl_gasoline_surrogate_323.xml",
            max_injections=2,
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
        npt.assert_allclose(np.linalg.norm(df.V), 0.003094822855555559)
        npt.assert_allclose(np.linalg.norm(df.p), 37096135.352912575)
        npt.assert_allclose(np.linalg.norm(df["T"]), 19946.248772614377)
        npt.assert_allclose(np.linalg.norm(df.rewards), 77.75784353525205)
        npt.assert_allclose(np.linalg.norm(df.mdot), 2.545584412271571)
        print(f"Wall time for DiscreteTwoZoneEngine with delay = {elapsed} seconds")

    def test_reactor_engine(self):
        """Does the ReactorEngine work as expected?"""

        # Initialize engine
        eng = engines.ReactorEngine(
            T0=self.T0,
            p0=self.p0,
            agent_steps=201,
            Tinj=300.0,
            minj=0.000026,
            dt=4e-6,
            rxnmech="dodecane_lu_nox.cti",
            small_negative_reward=-0.05,
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
        npt.assert_allclose(np.linalg.norm(df.V), 0.003094822855555559)
        npt.assert_allclose(np.linalg.norm(df.p), 52577638.641465, rtol=1e-5)
        npt.assert_allclose(np.linalg.norm(df["T"]), 18839.561176, rtol=1e-5)
        npt.assert_allclose(np.linalg.norm(df.rewards), 95.882818)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.23399999999999999)
        print(f"Wall time for ReactorEngine = {elapsed} seconds")

    def test_equilibrate_engine(self):
        """Does the EquilibrateEngine work as expected?"""

        # Initialize engine
        eng = engines.EquilibrateEngine(
            T0=self.T0,
            p0=self.p0,
            agent_steps=201,
            Tinj=300.0,
            minj=0.000026,
            rxnmech="dodecane_lu_nox.cti",
            small_negative_reward=-0.05,
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
        npt.assert_allclose(np.linalg.norm(df.V), 0.003094822855555559)
        npt.assert_allclose(np.linalg.norm(df.p), 58396518.33341535)
        npt.assert_allclose(np.linalg.norm(df["T"]), 17672.067301618787)
        npt.assert_allclose(np.linalg.norm(df.rewards), 99.783967)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.23399999999999999)
        print(f"Wall time for EquilibrateEngine = {elapsed} seconds")


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    unittest.main()
