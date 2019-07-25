# ========================================================================
#
# Imports
#
# ========================================================================
import os
import argparse
import pickle
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG, A2C, DQN
import utilities
import agents
import engine


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
    args = parser.parse_args()

    for k, fdir in enumerate(args.agents):

        # Setup
        fname = os.path.join(fdir, "agent")
        run_args = pickle.load(open(os.path.join(fdir, "args.pkl"), "rb"))

        # Initialize the engine
        T0, p0 = engine.calibrated_engine_ic()
        eng = engine.Engine(
            T0=T0,
            p0=p0,
            nsteps=run_args.nsteps,
            use_qdot=run_args.use_qdot,
            discrete_action=run_args.use_discrete,
        )

        env = DummyVecEnv([lambda: eng])
        if run_args.agent == "calibrated":
            agent = agents.CalibratedAgent(env)
            agent.learn()
        elif run_args.agent == "ddpg":
            eng.symmetrize_actions()
            env = DummyVecEnv([lambda: eng])
            agent = DDPG.load(fname, env=env)
        elif run_args.agent == "a2c":
            agent = A2C.load(fname, env=env)
        elif run_args.agent == "dqn":
            agent = DQN.load(fname, env=env)

        df, total_reward = utilities.evaluate_agent(env, agent)
        utilities.plot_df(env, df, idx=k, name=run_args.agent)

    utilities.save_plots("compare.pdf")
