# ========================================================================
#
# Imports
#
# ========================================================================
import argparse
import os
import sys
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import warnings
import pickle
import git
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import mprl.engines as engines
import mprl.agents as agents
import mprl.utilities as utilities
import mprl.inputs as inputs


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
    if isinstance(_locals["self"], PPO2):
        noutput = 10
        nint = int(
            np.ceil((_locals["total_timesteps"] / noutput) / _locals["self"].n_steps)
        )
        if _locals["self"].num_timesteps % (nint * _locals["self"].n_steps) == 0:
            print(f"""Checkpoint agent at step {_locals["self"].num_timesteps}""")
            _locals["self"].save(
                os.path.join(
                    _locals["self"].tensorboard_log,
                    f"""checkpoint_{_locals["self"].num_timesteps}.pkl""",
                )
            )

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
        "-f", "--fname", help="Input file name", type=str, required=True
    )
    args = parser.parse_args()

    # Read input parameters
    params = inputs.Input()
    params.from_toml(args.fname)

    # Setup
    start = time.time()
    np.random.seed(45473)
    logdir = os.path.dirname(os.path.abspath(args.fname))
    logname = os.path.join(logdir, "logger.csv")
    logs = pd.DataFrame(
        columns=["episode", "episode_step", "total_steps", "episode_reward"]
    )
    logs.to_csv(logname, index=False)
    repo = git.Repo(os.path.abspath(__file__), search_parent_directories=True)
    with open(os.path.join(logdir, "hash.txt"), "w") as f:
        f.write(f"hash: {repo.head.object.hexsha}\n")
    pickle.dump(args, open(os.path.join(logdir, "args.pkl"), "wb"))
    best_reward = -np.inf

    # Initialize the engine
    T0, p0 = engines.calibrated_engine_ic()
    eng_params = params.inputs["engine"]
    if eng_params["engine"].value == "reactor-engine":
        eng = engines.ReactorEngine(
            T0=T0,
            p0=p0,
            agent_steps=eng_params["nsteps"].value,
            mdot=eng_params["mdot"].value,
            max_minj=eng_params["max_minj"].value,
            max_injections=eng_params["max_injections"].value,
            injection_delay=eng_params["injection_delay"].value,
            negative_reward=eng_params["negative_reward"].value,
            fuel=eng_params["fuel"].value,
            rxnmech=eng_params["rxnmech"].value,
            observables=eng_params["observables"].value,
        )
    elif eng_params["engine"].value == "EQ-engine":
        eng = engines.EquilibrateEngine(
            T0=T0,
            p0=p0,
            agent_steps=eng_params["nsteps"].value,
            mdot=eng_params["mdot"].value,
            max_minj=eng_params["max_minj"].value,
            max_injections=eng_params["max_injections"].value,
            injection_delay=eng_params["injection_delay"].value,
            negative_reward=eng_params["negative_reward"].value,
            fuel=eng_params["fuel"].value,
            rxnmech=eng_params["rxnmech"].value,
            observables=eng_params["observables"].value,
        )
    elif eng_params["engine"].value == "twozone-engine":
        if eng_params["use_continuous"].value:
            eng = engines.ContinuousTwoZoneEngine(
                T0=T0,
                p0=p0,
                agent_steps=eng_params["nsteps"].value,
                negative_reward=eng_params["negative_reward"].value,
                fuel=eng_params["fuel"].value,
                rxnmech=eng_params["rxnmech"].value,
                use_qdot=eng_params["use_qdot"].value,
            )
        else:
            eng = engines.DiscreteTwoZoneEngine(
                T0=T0,
                p0=p0,
                agent_steps=eng_params["nsteps"].value,
                mdot=eng_params["mdot"].value,
                max_minj=eng_params["max_minj"].value,
                max_injections=eng_params["max_injections"].value,
                injection_delay=eng_params["injection_delay"].value,
                negative_reward=eng_params["negative_reward"].value,
                fuel=eng_params["fuel"].value,
                rxnmech=eng_params["rxnmech"].value,
                observables=eng_params["observables"].value,
            )

    # Create the agent and train
    agent_params = params.inputs["agent"]
    if agent_params["agent"].value == "calibrated":
        env = DummyVecEnv([lambda: eng])
        agent = agents.CalibratedAgent(env)
        agent.learn()
    elif agent_params["agent"].value == "exhaustive":
        env = DummyVecEnv([lambda: eng])
        agent = agents.ExhaustiveAgent(env)
        agent.learn()
    elif agent_params["agent"].value == "ppo":
        env = DummyVecEnv([lambda: eng])
        if agent_params["use_pretrained"].value is not None:
            agent = PPO2.load(
                os.path.join(agent_params["use_pretrained"].value, "agent"),
                env=env,
                reset_num_timesteps=False,
                n_steps=agent_params["update_nepisodes"].value
                * (eng_params["nsteps"].value - 1),
                tensorboard_log=logdir,
            )
        else:
            agent = PPO2(
                MlpPolicy,
                env,
                verbose=1,
                n_steps=agent_params["update_nepisodes"].value
                * (eng_params["nsteps"].value - 1),
                tensorboard_log=logdir,
            )
        agent.learn(
            total_timesteps=agent_params["number_episodes"].value
            * (eng_params["nsteps"].value - 1)
            * agent_params["nranks"].value,
            callback=callback,
        )

    # Save, evaluate, and plot the agent
    pfx = os.path.join(logdir, "agent")
    agent.save(pfx)
    env = DummyVecEnv([lambda: eng])
    df, total_reward = utilities.evaluate_agent(env, agent)

    df.to_csv(pfx + ".csv", index=False)
    utilities.plot_df(env, df, idx=0, name=agent_params["agent"].value)
    utilities.save_plots(pfx + ".pdf")

    # Plot the training history
    logs = pd.read_csv(logname)
    utilities.plot_training(logs, os.path.join(logdir, "logger.pdf"))

    # output timer
    end = time.time() - start
    print(f"Elapsed time {timedelta(seconds=end)} (or {end} seconds)")
