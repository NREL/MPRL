# ========================================================================
#
# Imports
#
# ========================================================================
import os
import sys
import shutil
import argparse
import numpy as np
import sherpa
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlpPolicy
from stable_baselines.sac.policies import MlpPolicy as sacMlpPolicy
from stable_baselines.ddpg.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
    AdaptiveParamNoiseSpec,
)
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import PPO1
from stable_baselines import SAC
import engine
import agents


# ========================================================================
#
# Classes
#
# ========================================================================
class Agent:
    def __init__(self, env):
        self.env = env


# ========================================================================
class A2CAgent(Agent):
    def __init__(self):
        Agent.__init__(self)

        self.parameters = [
            sherpa.Continuous("gamma", range=[0, 1]),
            sherpa.Continuous("learning_rate", range=[1e-8, 1e-1], scale="log"),
            sherpa.Ordinal(name="n_steps", range=[5, 10, 20, 40, 80]),
            sherpa.Choice(
                name="lr_schedule",
                range=[
                    "linear",
                    "constant",
                    "double_linear_con",
                    "middle_drop",
                    "double_middle_drop",
                ],
            ),
        ]

    def instantiate_agent(self, env, parameters):
        return A2C(
            MlpPolicy,
            env,
            verbose=1,
            learning_rate=parameters["learning_rate"],
            gamma=parameters["gamma"],
            n_steps=parameters["n_steps"],
            lr_schedule=parameters["lr_schedule"],
        )


# ========================================================================
class DDPGAgent(Agent):
    def __init__(self):
        Agent.__init__(self)

        self.parameters = [
            sherpa.Continuous("gamma", range=[0, 1]),
            sherpa.Continuous("tau", range=[0, 1]),
            sherpa.Ordinal(name="batch_size", range=[16, 32, 64, 128]),
            sherpa.Continuous("actor_lr", range=[1e-8, 1e-1], scale="log"),
            sherpa.Continuous("critic_lr", range=[1e-8, 1e-1], scale="log"),
            sherpa.Continuous("random_exploration", range=[0, 1]),
        ]

    def instantiate_agent(self, env, parameters):
        n_actions = env.action_space.shape[-1]
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions)
        )
        return DDPG(
            ddpgMlpPolicy,
            env,
            verbose=1,
            param_noise=param_noise,
            action_noise=action_noise,
            gamma=parameters["gamma"],
            tau=parameters["tau"],
            batch_size=parameters["batch_size"],
            actor_lr=parameters["actor_lr"],
            critic_lr=parameters["critic_lr"],
            random_exploration=parameters["random_exploration"],
        )


# ========================================================================
class PPOAgent(Agent):
    def __init__(self):
        Agent.__init__(self)

        self.parameters = [
            sherpa.Continuous("gamma", range=[0, 1]),
            sherpa.Continuous("clip_param", range=[0, 0.5]),
            sherpa.Continuous("lam", range=[0, 1]),
            sherpa.Continuous("optim_stepsize", range=[1e-8, 1e-1], scale="log"),
            sherpa.Ordinal(name="optim_batchsize", range=[16, 32, 64, 128]),
        ]

    def instantiate_agent(self, env, parameters):
        return PPO1(
            MlpPolicy,
            env,
            verbose=1,
            gamma=parameters["gamma"],
            clip_param=parameters["clip_param"],
            lam=parameters["lam"],
            optim_stepsize=parameters["optim_stepsize"],
            optim_batchsize=parameters["optim_batchsize"],
        )


# ========================================================================
class SACAgent(Agent):
    def __init__(self):
        Agent.__init__(self)

        self.parameters = [
            sherpa.Continuous("gamma", range=[0, 1]),
            sherpa.Continuous("tau", range=[0, 1]),
            sherpa.Continuous("learning_rate", range=[1e-8, 1e-1], scale="log"),
            sherpa.Ordinal(name="batch_size", range=[16, 32, 64, 128]),
            sherpa.Continuous("random_exploration", range=[0, 1]),
        ]

    def instantiate_agent(self, env, parameters):
        return SAC(
            sacMlpPolicy,
            env,
            verbose=1,
            gamma=parameters["gamma"],
            tau=parameters["tau"],
            learning_rate=parameters["learning_rate"],
            batch_size=parameters["batch_size"],
            random_exploration=parameters["random_exploration"],
        )


# ========================================================================
class Herd:
    """
    Herd for Sherpa
    """

    def __init__(self, agent_type, pop, project_dir):

        # Create the agent
        if agent_type == "A2C":
            self.agent = agents.A2CAgent()
        elif agent_type == "DDPG":
            self.agent = agents.DDPGAgent()
        elif agent_type == "PPO":
            self.agent = agents.PPOAgent()
        elif agent_type == "SAC":
            self.agent = agents.SACAgent()
        else:
            sys.exit(f"Unrecognized agent type {agent_type}")

        # Set an evolutionary algorithm for parameter search, enforce early stopping
        algorithm = sherpa.algorithms.PopulationBasedTraining(population_size=pop)
        rule = sherpa.algorithms.MedianStoppingRule(min_iterations=5, min_trials=1)
        self.study = sherpa.Study(
            self.agent.parameters,
            algorithm,
            lower_is_better=False,
            stopping_rule=rule,
            dashboard_port=9800,
        )

        self.logs_dir = os.path.join(project_dir, f"{agent_type}-logs")
        if os.path.exists(self.logs_dir):
            shutil.rmtree(self.logs_dir)
        os.makedirs(self.logs_dir)

    def study_the_population(self, env, n_epochs, total_timesteps):

        for tr_idx, trial in enumerate(self.study):

            # Create agent for each trial
            agent = self.agent.instantiate_agent(env, trial.parameters)

            for e_idx in range(n_epochs):

                # Train the agent
                agent.learn(total_timesteps=total_timesteps)

                # Get the total reward for this agent
                obs = env.reset()
                total_reward = 0.0
                for index in env.envs[0].history.index[1:]:
                    action, _ = agent.predict(obs)
                    obs, reward, done, info = env.step(action)
                    total_reward += reward[0]
                    if done:
                        break

                print(
                    f"Trial {tr_idx} (iteration {e_idx}): total reward = {total_reward}"
                )
                self.study.add_observation(
                    trial=trial, iteration=e_idx, objective=total_reward, context={}
                )
                if self.study.should_trial_stop(trial):
                    break

            self.study.finalize(trial)
            print(self.study.results)
            self.study.save(self.logs_dir)


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Hyperparameter search for an agent")
    parser.add_argument("-a", "--agent", help="Agent to tune", type=str, required=True)
    parser.add_argument("-p", "--pop", help="Population size", type=int, default=20)
    parser.add_argument(
        "-s",
        "--steps",
        help="Number of time steps for each agent",
        type=int,
        default=1e6,
    )
    parser.add_argument(
        "-e", "--epochs", help="Number of epochs per time step", type=int, default=100
    )
    parser.add_argument
    args = parser.parse_args()

    # Setup the environment
    T0 = 273.15 + 120
    p0 = 264_647.769_165_039_06
    engine = engine.Engine(T0=T0, p0=p0, nsteps=100)
    env = DummyVecEnv([lambda: engine])

    # Initialize the herd and study it
    herd = Herd(args.agent, args.pop, os.getcwd())
    herd.study_the_population(env, args.epochs, args.steps)
