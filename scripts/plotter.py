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
    parser = argparse.ArgumentParser(description="Train and evaluate an agent")
    parser.add_argument(
        "-a",
        "--agents",
        help="Folders containing agents to plot",
        type=str,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--engine_type",
        help="Engine type to use",
        type=str,
        default="twozone-engine",
        choices=["twozone-engine", "reactor-engine"],
    )
    args = parser.parse_args()

    for k, fdir in enumerate(args.agents):

        # Setup
        fname = os.path.join(fdir, "agent")
        run_args = pickle.load(open(os.path.join(fdir, "args.pkl"), "rb"))

        # Initialize the engine
        T0, p0 = engines.calibrated_engine_ic()
        if args.engine_type == "reactor-engine":
            eng = engines.ReactorEngine(T0=T0, p0=p0)
        elif args.engine_type == "twozone-engine":
            if args.use_continuous:
                eng = engines.ContinuousTwoZoneEngine(
                    T0=T0, p0=p0, nsteps=args.nsteps, use_qdot=args.use_qdot
                )
            else:
                eng = engines.DiscreteTwoZoneEngine(T0=T0, p0=p0, nsteps=args.nsteps)

        env = DummyVecEnv([lambda: eng])
        if run_args.agent == "calibrated":
            agent = agents.CalibratedAgent(env)
            agent.learn()
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
