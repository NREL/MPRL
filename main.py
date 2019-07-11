# ========================================================================
#
# Imports
#
# ========================================================================
import argparse
import os
import shutil
import numpy as np
import pandas as pd
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
# Functions
#
# ========================================================================
def callback(_locals, _globals):
    """
    Callback for agent
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global best_reward

    # After each episode, log the reward
    if _locals["done"]:
        df = pd.read_csv(logname)
        df.loc[len(df)] = [
            _locals["episodes"],
            _locals["episode_step"],
            _locals["total_steps"],
            _locals["episode_reward"],
        ]
        df.to_csv(logname, index=False)

        # save the agent if it is any good
        if _locals["episode_reward"] > best_reward:
            print("Saving new best agent")
            best_reward = _locals["episode_reward"]
            _locals["self"].save(os.path.join(logdir, "best_agent.pkl"))


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
    parser.add_argument(
        "--use_pretrained",
        help="Use a pretrained network as a starting point",
        action="store_true",
    )
    parser.add_argument
    args = parser.parse_args()

    # Setup
    start = time.time()
    nsteps = 100
    np.random.seed(45473)
    logdir = f"{args.agent}-logs"
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    logname = os.path.join(logdir, "logger.csv")
    logs = pd.DataFrame(
        columns=["episode", "episode_step", "total_steps", "episode_reward"]
    )
    logs.to_csv(logname, index=False)

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
        if args.use_pretrained:
            agent = DDPG.load(f"{args.agent}_pretrained")
            agent.set_env(env)
            _, best_reward = utilities.evaluate_agent(DummyVecEnv([lambda: eng]), agent)
        else:
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
        agent.learn(total_timesteps=args.steps, callback=callback)
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

    # Plot the training history
    logs = pd.read_csv(logname)
    utilities.plot_training(logs, os.path.join(logdir, "logger.pdf"))

    # output timer
    end = time.time() - start
    print(f"Elapsed time {timedelta(seconds=end)} (or {end} seconds)")
