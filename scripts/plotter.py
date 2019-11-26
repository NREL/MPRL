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
        "-c",
        "--checkpoints",
        help="Checkpoint numbers to plot",
        type=str,
        required=False,
        nargs="+",
    )
    parser.add_argument(
        "-s",
        "--nsteps",
        help="Total agent steps in a given episode",
        type=int,
        default=201,
    )
    parser.add_argument(
        "--engine_type",
        help="Engine type to use",
        type=str,
        choices=["twozone-engine", "reactor-engine", "EQ-engine"],
    )

    args = parser.parse_args()

    if args.checkpoints:
        mode = "checkpoints"
        iter_arg = args.checkpoints
    else:
        mode = "agents"
        iter_arg = args.agents

    for k, fiter in enumerate(iter_arg):

        if mode == "checkpoints":
            fdir = args.agents[0]
            fname = os.path.join(fdir, "checkpoint_" + fiter + ".pkl")
        else:
            fdir = fiter
            fname = os.path.join(fdir, "agent")

        run_args = pickle.load(open(os.path.join(fdir, "args.pkl"), "rb"))

        # Initialize the engine
        if args.engine_type:
            engine_type = args.engine_type
        else:
            engine_type = run_args.engine_type

        T0, p0 = engines.calibrated_engine_ic()
        if engine_type == "reactor-engine":
            eng = engines.ReactorEngine(
                T0=T0,
                p0=p0,
                agent_steps=args.nsteps,
                fuel=run_args.fuel,
                rxnmech=run_args.rxnmech,
                observables=run_args.observables,
            )
        elif engine_type == "EQ-engine":
            eng = engines.EquilibrateEngine(
                T0=T0,
                p0=p0,
                agent_steps=args.nsteps,
                fuel=run_args.fuel,
                rxnmech=run_args.rxnmech,
                observables=run_args.observables,
            )
        elif engine_type == "twozone-engine":
            if run_args.use_continuous:
                eng = engines.ContinuousTwoZoneEngine(
                    T0=T0,
                    p0=p0,
                    agent_steps=args.nsteps,
                    use_qdot=run_args.use_qdot,
                    fuel=run_args.fuel,
                    rxnmech=run_args.rxnmech,
                )
            else:
                eng = engines.DiscreteTwoZoneEngine(
                    T0=T0,
                    p0=p0,
                    agent_steps=args.nsteps,
                    fuel=run_args.fuel,
                    rxnmech=run_args.rxnmech,
                    observables=run_args.observables,
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
        if mode == "checkpoints":
            utilities.plot_df(env, df, idx=k, name=k)
        else:
            utilities.plot_df(env, df, idx=k, name=run_args.agent)

    utilities.save_plots("compare.pdf")
