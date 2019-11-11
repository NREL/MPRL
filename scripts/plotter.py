# ========================================================================
#
# Imports
#
# ========================================================================
import os
import sys
import argparse
import pickle
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG, A2C, DQN, PPO2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import mprl.engines as engines
import mprl.agents as agents
import mprl.utilities as utilities


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot agents")
    parser.add_argument(
        "-a",
        "--agents",
        help="Folders containing agents to plot",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-s", "--nsteps", help="Total steps in a given episode", type=int, default=201
    )
    args = parser.parse_args()

    for k, fdir in enumerate(args.agents):

        # Setup
        fname = os.path.join(fdir, "agent")
        run_args = pickle.load(open(os.path.join(fdir, "args.pkl"), "rb"))

        # Initialize the engine
        T0, p0 = engines.calibrated_engine_ic()
        if run_args.engine_type == "reactor-engine":
            eng = engines.ReactorEngine(
                T0=T0,
                p0=p0,
                agent_steps=args.nsteps,
                fuel=run_args.fuel,
                rxnmech=run_args.rxnmech,
            )
        elif run_args.engine_type == "twozone-engine":
            if run_args.use_continuous:
                eng = engines.ContinuousTwoZoneEngine(
                    T0=T0,
                    p0=p0,
                    nsteps=args.nsteps,
                    use_qdot=run_args.use_qdot,
                    fuel=run_args.fuel,
                    rxnmech=run_args.rxnmech,
                )
            else:
                eng = engines.DiscreteTwoZoneEngine(
                    T0=T0,
                    p0=p0,
                    nsteps=args.nsteps,
                    fuel=run_args.fuel,
                    rxnmech=run_args.rxnmech,
                )

        env = DummyVecEnv([lambda: eng])
        if run_args.agent == "calibrated":
            agent = agents.CalibratedAgent(env)
            agent.learn()
        elif run_args.agent == "exhaustive":
            agent = agents.ExhaustiveAgent(env)
            agent.load(fname, env)
        elif run_args.agent == "ddpg":
            eng.action.symmetrize_space()
            env = DummyVecEnv([lambda: eng])
            agent = DDPG.load(fname, env=env)
        elif run_args.agent == "a2c":
            agent = A2C.load(fname, env=env)
        elif run_args.agent == "ppo":
            agent = PPO2.load(fname, env=env)
        elif run_args.agent == "dqn":
            agent = DQN.load(fname, env=env)

        df, total_reward = utilities.evaluate_agent(env, agent)
        utilities.plot_df(env, df, idx=k, name=run_args.agent)

    utilities.save_plots("compare.pdf")
