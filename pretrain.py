import argparse
import numpy as np
import time
from datetime import timedelta
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as sacMlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlpPolicy
from stable_baselines.ddpg.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
    AdaptiveParamNoiseSpec,
)
from stable_baselines.gail import ExpertDataset
from stable_baselines import A2C
from stable_baselines import PPO1
from stable_baselines import SAC
from stable_baselines import TRPO
from stable_baselines import DDPG
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
    parser = argparse.ArgumentParser(description="Pretrain an agent")
    parser.add_argument(
        "-a",
        "--agent",
        help="Agent to pretrain",
        type=str,
        default="a2c",
        choices=["a2c", "ppo", "sac", "trpo", "ddpg"],
    )
    parser.add_argument(
        "-l", "--learning_rate", help="Learning rate", type=float, default=1e-3
    )
    parser.add_argument(
        "-e", "--epochs", help="Number of epochs", type=int, default=10000
    )
    parser.add_argument
    args = parser.parse_args()
    start = time.time()

    # Create the environment
    T0, p0 = engine.calibrated_engine_ic()
    eng = engine.Engine(T0=T0, p0=p0, nsteps=200)
    env = DummyVecEnv([lambda: eng])

    # Generate expert trajectories
    expert_agent = agents.CalibratedAgent(env)
    expert_agent.learn(use_qdot=False)
    traj = expert_agent.generate_expert_traj("calibrated.npz")
    expert_traj = ExpertDataset(
        expert_path="calibrated.npz", traj_limitation=1, batch_size=32
    )
    df, total_reward = utilities.evaluate_agent(env, expert_agent)
    utilities.plot_df(env, df, idx=0)

    # Create the agent
    if args.agent == "a2c":
        agent = A2C(MlpPolicy, env, verbose=1)

    elif args.agent == "ppo":
        agent = PPO1(MlpPolicy, env, verbose=1)

    elif args.agent == "trpo":
        agent = TRPO(MlpPolicy, env, verbose=1)

    elif args.agent == "sac":
        env.envs[0].symmetrize_actions()
        agent = SAC(sacMlpPolicy, env, verbose=1)

    elif args.agent == "ddpg":
        env.envs[0].symmetrize_actions()
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

    # Pretrain, save and plot the agent
    agent.pretrain(expert_traj, n_epochs=args.epochs, learning_rate=args.learning_rate)
    agent.save(f"{args.agent}_pretrained")
    df, total_reward = utilities.evaluate_agent(env, agent)
    utilities.plot_df(env, df, idx=1)
    utilities.save_plots(f"{args.agent}_pretrained.pdf")

    # output timer
    end = time.time() - start
    print(f"Elapsed time {timedelta(seconds=end)} (or {end} seconds)")
