# ========================================================================
#
# Imports
#
# ========================================================================
import os
import sys
import time
import copy
import pickle
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


def assert_similar_objects(orig, other, msg):
    """Test if two objects are the same but different in memory"""
    assert other is not orig, f"{msg}: objects are the same in memory"
    for v in dir(orig):
        assert hasattr(other, v), f"{msg}: copy does not contain {v}"
    assert orig == other, f"{msg}: Engines are not equal"


def assert_similar_step(orig, other):
    """Test if two engines step similarly"""
    orig.reset()
    other.reset()

    # Step the engines
    obs, reward, done, info = orig.step([0])
    other_obs, other_reward, other_done, other_info = other.step([0])

    # Compare
    npt.assert_allclose(np.linalg.norm(other_obs), np.linalg.norm(obs))
    npt.assert_allclose(other_reward, reward)
    assert other_done == done, "Dones are not equal"
    npt.assert_allclose(other_info["current_state"]["V"], info["current_state"]["V"])
    npt.assert_allclose(other_info["current_state"]["p"], info["current_state"]["p"])
    npt.assert_allclose(other_info["current_state"]["T"], info["current_state"]["T"])

    orig.reset()
    other.reset()


def assert_deepcopy_pickle_repr(orig):
    """Tests if an object can be deepcopied, pickled, and eval(repr)'ed."""

    # Deepcopy
    dc = copy.deepcopy(orig)
    assert_similar_objects(orig, dc, "Error in deepcopy")
    dc.reset()

    # Pickling
    fname = "test.pkl"
    with open(fname, "wb") as f:
        pickle.dump(orig, f)
    with open(fname, "rb") as f:
        pkl = pickle.load(f)
    assert_similar_objects(orig, pkl, "Error in pickling")
    pkl.reset()
    os.remove(fname)

    # Using repr
    rep = eval(f"engines.{repr(orig)}")
    assert_similar_objects(orig, rep, "Error in repr copying")
    rep.reset()

    # Step the engines and check
    assert_similar_step(orig, dc)
    assert_similar_step(orig, pkl)
    assert_similar_step(orig, rep)


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
        t0 = time.time()
        df, total_reward = utilities.evaluate_agent(env, agent)
        elapsed = time.time() - t0
        utilities.plot_df(env, df, idx=0, name="calibrated")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.002195212151)
        npt.assert_allclose(np.linalg.norm(df.p), 22012100.17143623)
        npt.assert_allclose(np.linalg.norm(df["T"]), 14210.47933497)
        npt.assert_allclose(np.linalg.norm(df.rewards), 104.47361754)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.04144044)
        npt.assert_allclose(np.linalg.norm(df.qdot), 97686.91574242)
        print(f"Wall time for CalibratedAgent = {elapsed} seconds")

    def test_exhaustive_agent(self):
        """Does the exhaustive agent work as expected?"""

        # Initialize engine
        eng = engines.DiscreteTwoZoneEngine(
            nsteps=101,
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
            nsteps=101,
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
        df.loc[cnt, variables] = [eng.current_state[k] for k in variables]
        df.loc[cnt, eng.action.actions] = 0
        df.loc[cnt, ["rewards"]] = [engines.get_reward(eng.current_state)]

        while not done:
            cnt += 1
            # Agent injects twice
            action = (
                [1]
                if (eng.current_state["ca"] == -10) or eng.current_state["ca"] == 10
                else [0]
            )
            obs, reward, done, info = env.step(action)
            df.loc[cnt, variables] = [info[0]["current_state"][k] for k in variables]
            df.loc[cnt, eng.action.actions] = eng.action.current
            df.loc[cnt, ["rewards"]] = reward

        elapsed = time.time() - t0

        utilities.plot_df(env, df, idx=2, name="discrete")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.002205916821815495)
        npt.assert_allclose(np.linalg.norm(df.p), 37091026.33424518)
        npt.assert_allclose(np.linalg.norm(df["T"]), 21272.66042475)
        npt.assert_allclose(np.linalg.norm(df.rewards), 155.27880220)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.14142136)
        print(f"Wall time for DiscreteTwoZoneEngine = {elapsed} seconds")

    def test_discrete_twozone_engine_with_delay(self):
        """Does the DiscreteTwoZoneEngine with injection delay work as expected?"""

        # Initialize engine
        eng = engines.DiscreteTwoZoneEngine(
            nsteps=101,
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
        df.loc[cnt, variables] = [eng.current_state[k] for k in variables]
        df.loc[cnt, eng.action.actions] = 0
        df.loc[cnt, ["rewards"]] = [engines.get_reward(eng.current_state)]

        while not done:
            cnt += 1
            # Agent tries to inject thrice, but is not allowed the second time
            action = (
                [1]
                if (eng.current_state["ca"] == -10)
                or eng.current_state["ca"] == 10
                or eng.current_state["ca"] == 16
                else [0]
            )
            obs, reward, done, info = env.step(action)
            df.loc[cnt, variables] = [info[0]["current_state"][k] for k in variables]
            df.loc[cnt, eng.action.actions] = eng.action.current
            df.loc[cnt, ["rewards"]] = reward

        elapsed = time.time() - t0

        utilities.plot_df(env, df, idx=5, name="DiscreteTwoZone (delay)")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.002205916821815495)
        npt.assert_allclose(np.linalg.norm(df.p), 35142241.61422163)
        npt.assert_allclose(np.linalg.norm(df["T"]), 20971.06692877)
        npt.assert_allclose(np.linalg.norm(df.rewards), 153.11736297)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.14142136)
        print(f"Wall time for DiscreteTwoZoneEngine with delay = {elapsed} seconds")

    def test_reactor_engine(self):
        """Does the ReactorEngine work as expected?"""

        # Initialize engine
        eng = engines.ReactorEngine(
            nsteps=101,
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
        df.loc[cnt, variables] = [eng.current_state[k] for k in variables]
        df.loc[cnt, eng.action.actions] = 0
        df.loc[cnt, ["rewards"]] = [engines.get_reward(eng.current_state)]

        while not done:
            cnt += 1
            # Agent tries to inject twice, but is not allowed the second time
            action = (
                [1]
                if (eng.current_state["ca"] == -10) or eng.current_state["ca"] == 10
                else [0]
            )
            obs, reward, done, info = env.step(action)
            df.loc[cnt, variables] = [info[0]["current_state"][k] for k in variables]
            df.loc[cnt, eng.action.actions] = eng.action.current
            df.loc[cnt, ["rewards"]] = reward

        elapsed = time.time() - t0

        utilities.plot_df(env, df, idx=3, name="reactor")

        # Test
        npt.assert_allclose(np.linalg.norm(df.V), 0.002205916821815495)
        npt.assert_allclose(np.linalg.norm(df.p), 35918721.97084988, rtol=1e-5)
        npt.assert_allclose(np.linalg.norm(df["T"]), 18047.51453986081, rtol=1e-5)
        npt.assert_allclose(np.linalg.norm(df.rewards), 157.92497130612531, rtol=1e-5)
        npt.assert_allclose(np.linalg.norm(df.mdot), 0.14142135623730953)
        print(f"Wall time for ReactorEngine = {elapsed} seconds")

    def test_equilibrate_engine(self):
        """Does the EquilibrateEngine work as expected?"""

        # Initialize engine
        eng = engines.EquilibrateEngine(
            nsteps=101,
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
        df.loc[cnt, variables] = [eng.current_state[k] for k in variables]
        df.loc[cnt, eng.action.actions] = 0
        df.loc[cnt, ["rewards"]] = [engines.get_reward(eng.current_state)]

        while not done:
            cnt += 1
            # Agent tries to inject twice, but is not allowed the second time
            action = (
                [1]
                if (eng.current_state["ca"] == -10) or eng.current_state["ca"] == 10
                else [0]
            )
            obs, reward, done, info = env.step(action)
            df.loc[cnt, variables] = [info[0]["current_state"][k] for k in variables]
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

    def test_discrete_twozone_engine_copies(self):
        """Do deepcopy, pickling and repr of DiscreteTwoZoneEngine work as expected?"""

        orig = engines.DiscreteTwoZoneEngine(
            nsteps=101, fuel="dodecane", rxnmech="dodecane_lu_nox.cti"
        )
        assert_deepcopy_pickle_repr(orig)

    def test_equilibrate_engine_copies(self):
        """Do deepcopy, pickling and repr of EquilibrateEngine work as expected?"""

        orig = engines.EquilibrateEngine(
            nsteps=101,
            Tinj=300.0,
            rxnmech="dodecane_lu_nox.cti",
            mdot=0.01,
            max_minj=5e-5,
            negative_reward=-0.05,
        )
        assert_deepcopy_pickle_repr(orig)

    def test_reactor_engine_copies(self):
        """Do deepcopy, pickling and repr of ReactorEngine work as expected?"""

        orig = engines.ReactorEngine(
            nsteps=101,
            Tinj=300.0,
            rxnmech="dodecane_lu_nox.cti",
            mdot=0.1,
            max_minj=5e-5,
            negative_reward=-0.05,
        )
        assert_deepcopy_pickle_repr(orig)


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    unittest.main()
