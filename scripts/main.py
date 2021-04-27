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
from datetime import timedelta
import warnings
import pickle
import git
from io import BytesIO
from PIL import Image
from matplotlib import cm
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import mprl.engines as engines
import mprl.agents as agents
import mprl.utilities as utilities
import mprl.inputs as inputs
import mprl.reward as rw


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

        # Get the actions of the last episode
        idx = np.argwhere(_locals["masks"])[-1][0]
        actions = np.array(_locals["actions"][idx:]).reshape(-1, 1)

        if not os.path.exists(aname):
            np.savez(aname, actions=actions)
        else:
            npzf = np.load(aname)
            actions = np.hstack((npzf["actions"], actions))
            np.savez(aname, actions=actions)

        img = Image.fromarray(np.uint8(cm.viridis(actions * 1.0) * 255))
        with BytesIO() as output:
            img.save(output, "PNG")
            imgb = output.getvalue()

        # Write some custom stuff to tensorboard
        writer = _locals["writer"]
        eng = _locals["self"].env.envs[0]
        summ = tf.Summary(
            value=[
                tf.Summary.Value(tag=f"rewards/w_{k}", simple_value=v)
                for k, v in eng.info["reward_weights"].items()
            ]
            + [
                tf.Summary.Value(tag=f"rewards/r_{k}", simple_value=v)
                for k, v in eng.info["returns"].items()
            ]
            + [
                tf.Summary.Value(
                    tag=f"actions",
                    image=tf.Summary.Image(
                        height=actions.shape[0],
                        width=actions.shape[1],
                        colorspace=4,
                        encoded_image_string=imgb,
                    ),
                )
            ]
        )
        writer.add_summary(summ, _locals["self"].num_timesteps)

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
    np.random.seed(454733)
    logdir = os.path.dirname(os.path.abspath(args.fname))
    logname = os.path.join(logdir, "logger.csv")
    logs = pd.DataFrame(
        columns=["episode", "episode_step", "total_steps", "episode_reward"]
    )
    logs.to_csv(logname, index=False)
    aname = os.path.join(logdir, "actions.npz")
    if os.path.exists(aname):
        os.remove(aname)
    repo = git.Repo(os.path.abspath(__file__), search_parent_directories=True)
    with open(os.path.join(logdir, "hash.txt"), "w") as f:
        f.write(f"hash: {repo.head.object.hexsha}\n")
    pickle.dump(args, open(os.path.join(logdir, "args.pkl"), "wb"))
    best_reward = -np.inf

    # Initialize the reward
    rwd_params = params.inputs["reward"]
    reward = rw.Reward(
        names=rwd_params["names"].value,
        norms=rwd_params["norms"].value,
        weights=rwd_params["weights"].value,
        negative_reward=rwd_params["negative_reward"].value,
        EOC_reward=rwd_params["EOC_reward"].value,
        randomize=rwd_params["randomize"].value,
        random_updates=rwd_params["random_updates"].value,
    )

    # Initialize the engine
    eng_params = params.inputs["engine"]
    if eng_params["engine"].value == "reactor-engine":
        eng = engines.ReactorEngine(
            Tinj=eng_params["Tinj"].value,
            nsteps=eng_params["nsteps"].value,
            mdot=eng_params["mdot"].value,
            max_minj=eng_params["max_minj"].value,
            injection_delay=eng_params["injection_delay"].value,
            max_pressure=eng_params["max_pressure"].value,
            ename=eng_params["ename"].value,
            reward=reward,
            fuel=eng_params["fuel"].value,
            rxnmech=eng_params["rxnmech"].value,
            observables=eng_params["observables"].value,
            use_qdot=eng_params["use_qdot"].value,
        )
    elif eng_params["engine"].value == "EQ-engine":
        eng = engines.EquilibrateEngine(
            Tinj=eng_params["Tinj"].value,
            nsteps=eng_params["nsteps"].value,
            mdot=eng_params["mdot"].value,
            max_minj=eng_params["max_minj"].value,
            injection_delay=eng_params["injection_delay"].value,
            max_pressure=eng_params["max_pressure"].value,
            ename=eng_params["ename"].value,
            reward=reward,
            fuel=eng_params["fuel"].value,
            rxnmech=eng_params["rxnmech"].value,
            observables=eng_params["observables"].value,
            use_qdot=eng_params["use_qdot"].value,
        )
    elif eng_params["engine"].value == "twozone-engine":
        if eng_params["use_continuous"].value:
            eng = engines.ContinuousTwoZoneEngine(
                nsteps=eng_params["nsteps"].value,
                max_pressure=eng_params["max_pressure"].value,
                ename=eng_params["ename"].value,
                reward=reward,
                fuel=eng_params["fuel"].value,
                twozone_phi=eng_params["twozone_phi"].value,
                rxnmech=eng_params["rxnmech"].value,
                use_qdot=eng_params["use_qdot"].value,
            )
        else:
            eng = engines.DiscreteTwoZoneEngine(
                nsteps=eng_params["nsteps"].value,
                mdot=eng_params["mdot"].value,
                max_minj=eng_params["max_minj"].value,
                injection_delay=eng_params["injection_delay"].value,
                max_pressure=eng_params["max_pressure"].value,
                ename=eng_params["ename"].value,
                reward=reward,
                fuel=eng_params["fuel"].value,
                twozone_phi=eng_params["twozone_phi"].value,
                rxnmech=eng_params["rxnmech"].value,
                observables=eng_params["observables"].value,
                use_qdot=eng_params["use_qdot"].value,
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
        agent.learn(nranks=agent_params["nranks"].value)
    elif agent_params["agent"].value == "ppo":
        env = DummyVecEnv([lambda: eng])
        if agent_params["pretrained_agent"].value is not None:
            agent = PPO2.load(
                agent_params["pretrained_agent"].value,
                env=env,
                reset_num_timesteps=False,
                n_steps=agent_params["update_nepisodes"].value
                * (eng_params["nsteps"].value - 1),
                learning_rate=agent_params["learning_rate"].value,
                gamma=agent_params["gamma"].value,
                tensorboard_log=logdir,
            )
        else:
            agent = PPO2(
                MlpPolicy,
                env,
                verbose=1,
                n_steps=agent_params["update_nepisodes"].value
                * (eng_params["nsteps"].value - 1),
                learning_rate=agent_params["learning_rate"].value,
                gamma=agent_params["gamma"].value,
                tensorboard_log=logdir,
            )
        agent.learn(
            total_timesteps=agent_params["number_episodes"].value
            * (eng_params["nsteps"].value - 1)
            * agent_params["nranks"].value,
            callback=callback,
        )
    elif agent_params["agent"].value == "manual":
        env = DummyVecEnv([lambda: eng])
        agent = agents.ManualAgent(env)
        agent.learn(agent_params["injection_cas"].value, agent_params["qdot_cas"].value)

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
