# ========================================================================
#
# Imports
#
# ========================================================================
import os
import sys
import sherpa
from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
    AdaptiveParamNoiseSpec,
)
from stable_baselines import DDPG
from stable_baselines import A2C


# ========================================================================
#
# Functions
#
# ========================================================================
class SherpaAgent:
    def __init__(self, agent_type):

        self.agent_type = agent_type

        # A2C Parameters
        if self.agent_type == "A2C":
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

        else:
            sys.exit(f"Unrecognized agent type {agent_type}")

    def instantiate_agent(self, env, parameters):
        if self.agent_type == "A2C":
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
#
# Classes
#
# ========================================================================
class Herd:
    """
    Herd for Sherpa
    """

    def __init__(self, agent_type, pop, project_dir):

        self.agent = SherpaAgent(agent_type)

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
                    total_reward += reward
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
            # Save study
            self.study.save(self.logs_dir)
