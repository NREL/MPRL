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
import warnings
import pickle
from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as dqnMlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.ddpg.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
    AdaptiveParamNoiseSpec,
)
from stable_baselines import DDPG, A2C, DQN
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
    done = False
    if isinstance(_locals["self"], DDPG):
        if _locals["done"]:
            done = True
            info = [
                _locals["episodes"],
                _locals["episode_step"],
                _locals["total_steps"],
                _locals["episode_reward"],
            ]

    elif isinstance(_locals["self"], A2C):
        warnings.warn("Callback not implemented for this agent")
        # print(_locals, _locals["self"].episode_reward)
        # if _locals["runner"].dones[-1]:
        #     done = True
        #     episodes = 10
        #     print(episodes)

    elif isinstance(_locals["self"], DQN):
        try:
            done = _locals["done"]
        except KeyError:
            pass

        if done:
            info = [
                _locals["num_episodes"] - 1,
                _locals["info"]["internals"].name,
                _locals["_"],
                _locals["episode_rewards"][-2],
            ]

    else:
        warnings.warn("Callback not implemented for this agent")

    if done:
        df = pd.read_csv(logname)
        df.loc[len(df)] = info
        df.to_csv(logname, index=False)

        # save the agent if it is any good
        if df.episode_reward.iloc[-1] > best_reward:
            print("Saving new best agent")
            best_reward = df.episode_reward.iloc[-1]
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
        choices=["calibrated", "ddpg", "a2c", "dqn"],
    )
    parser.add_argument(
        "-t",
        "--total_timesteps",
        help="Total number of steps for training",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-s", "--nsteps", help="Total steps in a given episode", type=int, default=201
    )
    parser.add_argument("--nranks", help="Number of MPI ranks", type=int, default=1)
    parser.add_argument(
        "--use_pretrained",
        help="Use a pretrained network as a starting point",
        action="store_true",
    )
    parser.add_argument(
        "--use_qdot", help="Use a Qdot as an action", action="store_true"
    )
    parser.add_argument(
        "--use_discrete", help="Use a discrete action space", action="store_true"
    )
    args = parser.parse_args()

    # Setup
    start = time.time()
    np.random.seed(45473)
    logdir = f"{args.agent}"
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    logname = os.path.join(logdir, "logger.csv")
    logs = pd.DataFrame(
        columns=["episode", "episode_step", "total_steps", "episode_reward"]
    )
    logs.to_csv(logname, index=False)
    pickle.dump(args, open(os.path.join(logdir, "args.pkl"), "wb"))
    best_reward = -np.inf

    # Initialize the engine
    T0, p0 = engine.calibrated_engine_ic()
    eng = engine.Engine(
        T0=T0,
        p0=p0,
        nsteps=args.nsteps,
        use_qdot=args.use_qdot,
        discrete_action=args.use_discrete,
    )

    # Create the agent and train
    if args.agent == "calibrated":
        env = DummyVecEnv([lambda: eng])
        agent = agents.CalibratedAgent(env)
        agent.learn()
    elif args.agent == "ddpg":
        eng.symmetrize_actions()
        env = DummyVecEnv([lambda: eng])
        if args.use_pretrained:
            agent = DDPG.load(os.path.join(f"{args.agent}-pretrained", "agent"))
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
        agent.learn(total_timesteps=args.total_timesteps, callback=callback)
    elif args.agent == "a2c":
        env = SubprocVecEnv([lambda: eng for i in range(args.nranks)])
        agent = A2C(MlpPolicy, env, verbose=1, n_steps=1)
        agent.learn(total_timesteps=args.total_timesteps, callback=callback)
    elif args.agent == "dqn":
        env = DummyVecEnv([lambda: eng])
        agent = DQN(dqnMlpPolicy, env, verbose=1)
        agent.learn(total_timesteps=args.total_timesteps, callback=callback)

    # Save, evaluate, and plot the agent
    pfx = os.path.join(logdir, "agent")
    agent.save(pfx)
    env = DummyVecEnv([lambda: eng])
    df, total_reward = utilities.evaluate_agent(env, agent)
    df.to_csv(pfx + ".csv", index=False)
    utilities.plot_df(env, df, idx=0, name=args.agent)
    utilities.save_plots(pfx + ".pdf")

    # Plot the training history
    logs = pd.read_csv(logname)
    utilities.plot_training(logs, os.path.join(logdir, "logger.pdf"))

    # output timer
    end = time.time() - start
    print(f"Elapsed time {timedelta(seconds=end)} (or {end} seconds)")
