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
from stable_baselines import PPO2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import mprl.engines as engines
import mprl.agents as agents
import mprl.utilities as utilities
import mprl.inputs as inputs
import mprl.reward as rw


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot agents")
    parser.add_argument(
        "-a", "--agents", help="Agents to plot", type=str, required=True, nargs="+"
    )
    parser.add_argument(
        "-l", "--labels", help="Labels for plot", type=str, nargs="+", default=None
    )
    parser.add_argument(
        "-w",
        "--weights",
        help="Weights for rewards",
        type=float,
        nargs="+",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--legends",
        help="Figure titles where legend should appear",
        type=str,
        nargs="+",
        default=["p"],
    )
    args = parser.parse_args()

    for k, fname in enumerate(args.agents):
        fdir = os.path.dirname(os.path.abspath(fname))
        run_args = pickle.load(open(os.path.join(fdir, "args.pkl"), "rb"))

        # Read input parameters
        params = inputs.Input()
        params.from_toml(os.path.join(fdir, os.path.basename(run_args.fname)))

        # Initialize the reward
        rwd_params = params.inputs["reward"]
        if args.weights is None:
            weights = rwd_params["weights"].value
            randomize = rwd_params["randomize"].value
        else:
            weights = args.weights[k]
            randomize = False
            if len(weights) != len(rwd_params["weights"].value):
                sys.exit("Wrong weights input length")
        reward = rw.Reward(
            names=rwd_params["names"].value,
            norms=rwd_params["norms"].value,
            weights=weights,
            negative_reward=rwd_params["negative_reward"].value,
            EOC_reward=rwd_params["EOC_reward"].value,
            randomize=randomize,
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
            )
        elif eng_params["engine"].value == "twozone-engine":
            if eng_params["use_continuous"].value:
                eng = engines.ContinuousTwoZoneEngine(
                    nsteps=eng_params["nsteps"].value,
                    max_pressure=eng_params["max_pressure"].value,
                    ename=eng_params["ename"].value,
                    reward=reward,
                    fuel=eng_params["fuel"].value,
                    rxnmech=eng_params["rxnmech"].value,
                    use_qdot=eng_params["use_qdot"].value,
                    twozone_phi=eng_params["twozone_phi"].value,
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
                    rxnmech=eng_params["rxnmech"].value,
                    observables=eng_params["observables"].value,
                    twozone_phi=eng_params["twozone_phi"].value,
                )

        env = DummyVecEnv([lambda: eng])
        agent_params = params.inputs["agent"]
        if agent_params["agent"].value == "calibrated":
            agent = agents.CalibratedAgent(env)
            agent.learn()
        elif agent_params["agent"].value == "exhaustive":
            agent = agents.ExhaustiveAgent(env)
            agent.load(fname, env)
        elif agent_params["agent"].value == "ppo":
            agent = PPO2.load(fname, env=env)

        df, total_reward = utilities.evaluate_agent(env, agent)
        print(f"The total reward for {fname} is {total_reward}.")
        print(f"The total work for {fname} is {df.work.iloc[-1]}.")
        if "nox" in df.columns:
            print(f"The EOC NOx for {fname} is {df.nox.iloc[-1]}.")
        name = agent_params["agent"].value if args.labels is None else args.labels[k]
        utilities.plot_df(env, df, idx=k, name=name, plot_exp=False)

    utilities.save_plots("compare.pdf", legends=args.legends)
