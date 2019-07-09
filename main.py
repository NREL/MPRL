# ========================================================================
#
# Imports
#
# ========================================================================
import argparse
import numpy as np
import time
from datetime import timedelta
from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.ddpg.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
    AdaptiveParamNoiseSpec,
)
from stable_baselines import DDPG
from stable_baselines import A2C
import engine
import agents
import utilities


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
        "--agent",
        help="Agent to train and evaluate",
        type=str,
        default="calibrated",
        choices=["calibrated", "a2c", "ddpg"],
    )
    parser.add_argument(
        "-s",
        "--steps",
        help="Total number of steps for training",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-n", "--nranks", help="Number of MPI ranks", type=int, default=1
    )
    parser.add_argument
    args = parser.parse_args()

    # Setup
    start = time.time()
    nsteps = 100
    np.random.seed(45473)

    # Initialize the engine
    T0, p0 = engine.calibrated_engine_ic()
    eng = engine.Engine(T0=T0, p0=p0, nsteps=nsteps)

    # Create the agent and train
    if args.agent == "calibrated":
        env = DummyVecEnv([lambda: eng])
        agent = agents.CalibratedAgent(env)
        agent.learn(use_qdot=True)
    elif args.agent == "ddpg":
        eng.symmetrize_actions()
        env = DummyVecEnv([lambda: eng])
        n_actions = env.action_space.shape[-1]
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions)
        )
        agent = DDPG(
            ddpgMlpPolicy,
            env,
            verbose=1,
            param_noise=param_noise,
            action_noise=action_noise,
        )
        agent.learn(total_timesteps=args.steps)
    elif args.agent == "a2c":
        env = SubprocVecEnv([lambda: eng for i in range(args.nranks)])
        agent = A2C(MlpPolicy, env, verbose=1)
        agent.learn(total_timesteps=args.steps)

    # Save, evaluate, and plot the agent
    agent.save(args.agent)
    df, total_reward = utilities.evaluate_agent(DummyVecEnv([lambda: eng]), agent)
    df.to_csv(f"{args.agent}.csv", index=False)
    utilities.plot_df(env, df, idx=0)
    utilities.save_plots(f"{args.agent}.pdf")

    # output timer
    end = time.time() - start
    print(f"Elapsed time {timedelta(seconds=end)} (or {end} seconds)")
