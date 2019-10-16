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
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import PPO2
from stable_baselines import SAC
from stable_baselines import DQN

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import mprl.engines as engines


# ========================================================================
#
# Classes
#
# ========================================================================
class Agent:
    def __init__(self):
        self.sherpa_parameters = {
            "gamma": sherpa.Continuous("gamma", range=[0.8, 1]),
            "ent_coef": sherpa.Continuous("ent_coef", range=[1e-4, 1e-1], scale="log"),
            "vf_coef": sherpa.Continuous("vf_coef", range=[0, 1]),
            "max_grad_norm": sherpa.Continuous("max_grad_norm", range=[0, 1]),
            "lam": sherpa.Continuous("lam", range=[0.5, 1.0]),
            "nminibatches": sherpa.Ordinal("nminibatches", range=[2, 4, 8, 16]),
            "noptepochs": sherpa.Ordinal("noptepochs", range=[2, 4, 8, 16]),
            "cliprange": sherpa.Continuous("cliprange", range=[0.1, 0.9]),
            "tau": sherpa.Continuous("tau", range=[0, 0.1]),
            "random_exploration": sherpa.Continuous("random_exploration", range=[0, 1]),
            "exploration_fraction": sherpa.Continuous(
                "exploration_fraction", range=[0.5, 0.1]
            ),
            "exploration_final_eps": sherpa.Continuous(
                "exploration_final_eps", range=[0.02, 0.0001]
            ),
            "clip_param": sherpa.Continuous("clip_param", range=[0, 0.5]),
            "n_steps": sherpa.Ordinal("n_steps", range=[16, 32, 64, 128]),
            "batch_size": sherpa.Ordinal("batch_size", range=[32, 64, 128]),
            "buffer_size": sherpa.Ordinal("buffer_size", range=[1000, 10000]),
            "n_nodes": sherpa.Ordinal("n_nodes", range=[8, 16, 32, 64]),
            "prioritized_replay": sherpa.Ordinal(
                "prioritized_replay", range=[True, False]
            ),
            "layer_norm": sherpa.Ordinal("layer_norm", range=[True, False]),
            "DQNpolicy": sherpa.Choice("DQNpolicy", range=["MlpPolicy", "LnMlpPolicy"]),
            "optim_batchsize": (
                sherpa.Ordinal("optim_batchsize", range=[32, 64, 128]),
            ),
            "learning_rate": sherpa.Continuous(
                "learning_rate", range=[1e-5, 1e-3], scale="log"
            ),
            "actor_lr": (
                sherpa.Continuous("actor_lr", range=[1e-8, 1e-1], scale="log"),
            ),
            "critic_lr": (
                sherpa.Continuous("critic_lr", range=[1e-8, 1e-1], scale="log"),
            ),
            "optim_stepsize": (
                sherpa.Continuous("optim_stepsize", range=[1e-8, 1e-1], scale="log"),
            ),
            "lr_schedule": sherpa.Choice(
                "lr_schedule",
                range=[
                    "linear",
                    "constant",
                    "double_linear_con",
                    "middle_drop",
                    "double_middle_drop",
                ],
            ),
        }


# ========================================================================
class DQNAgent(Agent):
    def __init__(self):
        Agent.__init__(self)

        parameter_names = [
            "gamma",
            "learning_rate",
            "DQNpolicy",
            "exploration_fraction",
            "exploration_final_eps",
        ]
        self.parameters = [self.sherpa_parameters[x] for x in parameter_names]

    def instantiate_agent(self, env, parameters):
        return DQN(
            parameters["DQNpolicy"],
            env,
            verbose=1,
            gamma=parameters["gamma"],
            exploration_fraction=parameters["exploration_fraction"],
            exploration_final_eps=parameters["exploration_final_eps"],
            learning_rate=parameters["learning_rate"],
            buffer_size=10000,
            batch_size=128,
            prioritized_replay=True,
        )


# ========================================================================
class A2CAgent(Agent):
    def __init__(self):
        Agent.__init__(self)

        parameter_names = ["gamma", "learning_rate", "n_steps", "lr_schedule"]
        self.parameters = [self.sherpa_parameters[x] for x in parameter_names]

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

        parameter_names = [
            "gamma",
            "tau",
            "batch_size",
            "actor_lr",
            "critic_lr",
            "random_exploration",
        ]
        self.parameters = [self.sherpa_parameters[x] for x in parameter_names]

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

        parameter_names = [
            "gamma",
            "ent_coef",
            "learning_rate",
            "vf_coef",
            "max_grad_norm",
            "lam",
            "nminibatches",
            "noptepochs",
            "cliprange",
            "n_steps",
        ]
        self.parameters = [self.sherpa_parameters[x] for x in parameter_names]

    def instantiate_agent(self, env, parameters):
        return PPO2(
            MlpPolicy,
            env,
            verbose=1,
            gamma=parameters["gamma"],
            ent_coef=parameters["ent_coef"],
            learning_rate=parameters["learning_rate"],
            vf_coef=parameters["vf_coef"],
            max_grad_norm=parameters["max_grad_norm"],
            lam=parameters["lam"],
            nminibatches=parameters["nminibatches"],
            noptepochs=parameters["noptepochs"],
            cliprange_vf=parameters["cliprange"],
            n_steps=parameters["n_steps"],
        )


# ========================================================================
class SACAgent(Agent):
    def __init__(self):
        Agent.__init__(self)

        parameter_names = [
            "gamma",
            "tau",
            "learning_rate",
            "batch_size",
            "random_exploration",
        ]
        self.parameters = [self.sherpa_parameters[x] for x in parameter_names]

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
            self.agent = A2CAgent()
        elif agent_type == "DDPG":
            self.agent = DDPGAgent()
        elif agent_type == "PPO" or agent_type == "ppo":
            self.agent = PPOAgent()
        elif agent_type == "SAC":
            self.agent = SACAgent()
        elif agent_type == "DQN" or agent_type == "dqn":
            self.agent = DQNAgent()
        else:
            sys.exit(f"Unrecognized agent type {agent_type}")

        # Set an evolutionary algorithm for parameter search, enforce early stopping
        algorithm = sherpa.algorithms.PopulationBasedTraining(population_size=pop)
        rule = sherpa.algorithms.MedianStoppingRule(min_iterations=5, min_trials=10)
        self.study = sherpa.Study(
            self.agent.parameters,
            algorithm,
            lower_is_better=False,
            stopping_rule=rule,
            dashboard_port=None,
        )

        self.logs_dir = os.path.join(project_dir, f"{agent_type}-tuning")
        if os.path.exists(self.logs_dir):
            shutil.rmtree(self.logs_dir)
        os.makedirs(self.logs_dir)

    def study_the_population(self, env, n_epochs, steps_per_epoch):

        for tr_idx, trial in enumerate(self.study):

            # Create agent for each trial
            agent = self.agent.instantiate_agent(env, trial.parameters)

            for e_idx in range(n_epochs):
                obs = env.reset()

                # Train the agent
                agent.learn(
                    total_timesteps=int(steps_per_epoch), reset_num_timesteps=False
                )

                # Get the total reward for this agent. Only care about
                # the zeroth environment because all the environments
                # start the same
                obs = env.reset()
                total_reward = 0.0
                print("Evaluating agent ...")
                done = [False]
                while not done[0]:
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env.step(action)
                    total_reward += reward[0]

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
        "--steps_per_epoch",
        help="Number of steps per epoch for each agent",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "-e", "--epochs", help="Number of epochs", type=int, default=100
    )
    parser.add_argument(
        "-n", "--nranks", help="Number of MPI ranks", type=int, default=1
    )
    parser.add_argument(
        "-nep",
        help="Total number of episodes to train in each epoch",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--engine_type",
        help="Engine type to use",
        type=str,
        default="twozone-engine",
        choices=["twozone-engine", "reactor-engine"],
    )
    parser.add_argument(
        "--fuel",
        help="Fuel to use",
        type=str,
        default="dodecane",
        choices=["dodecane", "PRF100", "PRF85"],
    )
    parser.add_argument(
        "--rxnmech",
        help="Reaction mechanism to use",
        type=str,
        default="dodecane_lu_nox.cti",
        choices=[
            "dodecane_lu_nox.cti",
            "dodecane_mars.cti",
            "dodecane_lu.cti",
            "llnl_gasoline_surrogate_323.xml",
        ],
    )
    parser.add_argument
    args = parser.parse_args()

    # Initialize the engine
    T0, p0 = engines.calibrated_engine_ic()
    if args.engine_type == "reactor-engine":
        eng = engines.ReactorEngine(T0=T0, p0=p0, fuel=args.fuel, rxnmech=args.rxnmech)
    elif args.engine_type == "twozone-engine":
        if args.use_continuous:
            eng = engines.ContinuousTwoZoneEngine(
                T0=T0,
                p0=p0,
                nsteps=args.nsteps,
                use_qdot=args.use_qdot,
                fuel=args.fuel,
                rxnmech=args.rxnmech,
            )
        else:
            eng = engines.DiscreteTwoZoneEngine(
                T0=T0, p0=p0, nsteps=args.nsteps, fuel=args.fuel, rxnmech=args.rxnmech
            )
    env = DummyVecEnv([lambda: eng])

    # Initialize the herd and study it
    herd = Herd(args.agent, args.pop, os.getcwd())
    herd.study_the_population(
        env, args.epochs, args.steps_per_epoch * args.nep * args.nranks
    )
