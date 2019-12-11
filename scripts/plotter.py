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
    args = parser.parse_args()

    for k, fname in enumerate(args.agents):
        fdir = os.path.dirname(os.path.abspath(fname))
        run_args = pickle.load(open(os.path.join(fdir, "args.pkl"), "rb"))

        # Read input parameters
        params = inputs.Input()
        params.from_toml(os.path.join(fdir, os.path.basename(run_args.fname)))

        # Initialize the engine
        eng_params = params.inputs["engine"]
        if eng_params["engine"].value == "reactor-engine":
            eng = engines.ReactorEngine(
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
                    agent_steps=eng_params["nsteps"].value,
                    negative_reward=eng_params["negative_reward"].value,
                    fuel=eng_params["fuel"].value,
                    rxnmech=eng_params["rxnmech"].value,
                    use_qdot=eng_params["use_qdot"].value,
                )
            else:
                eng = engines.DiscreteTwoZoneEngine(
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
        utilities.plot_df(
            env, df, idx=k, name=agent_params["agent"].value, plot_exp=False
        )

    utilities.save_plots("compare.pdf")
