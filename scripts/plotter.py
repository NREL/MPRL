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
        T0, p0 = engines.calibrated_engine_ic()
        eng_params = params.input["engine"]
        if eng_params["engine"] == "reactor-engine":
            eng = engines.ReactorEngine(
                T0=T0,
                p0=p0,
                agent_steps=eng_params["nsteps"],
                mdot=eng_params["mdot"],
                max_minj=eng_params["max_minj"],
                max_injections=eng_params["max_injections"],
                injection_delay=eng_params["injection_delay"],
                small_negative_reward=eng_params["small_negative_reward"],
                fuel=eng_params["fuel"],
                rxnmech=eng_params["rxnmech"],
                observables=eng_params["observables"],
            )
        elif eng_params["engine"] == "EQ-engine":
            eng = engines.EquilibrateEngine(
                T0=T0,
                p0=p0,
                agent_steps=eng_params["nsteps"],
                mdot=eng_params["mdot"],
                max_minj=eng_params["max_minj"],
                max_injections=eng_params["max_injections"],
                injection_delay=eng_params["injection_delay"],
                small_negative_reward=eng_params["small_negative_reward"],
                fuel=eng_params["fuel"],
                rxnmech=eng_params["rxnmech"],
                observables=eng_params["observables"],
            )
        elif eng_params["engine"] == "twozone-engine":
            if eng_params["use_continuous"]:
                eng = engines.ContinuousTwoZoneEngine(
                    T0=T0,
                    p0=p0,
                    agent_steps=eng_params["nsteps"],
                    small_negative_reward=eng_params["small_negative_reward"],
                    fuel=eng_params["fuel"],
                    rxnmech=eng_params["rxnmech"],
                    use_qdot=eng_params["use_qdot"],
                )
            else:
                eng = engines.DiscreteTwoZoneEngine(
                    T0=T0,
                    p0=p0,
                    agent_steps=eng_params["nsteps"],
                    mdot=eng_params["mdot"],
                    max_minj=eng_params["max_minj"],
                    max_injections=eng_params["max_injections"],
                    injection_delay=eng_params["injection_delay"],
                    small_negative_reward=eng_params["small_negative_reward"],
                    fuel=eng_params["fuel"],
                    rxnmech=eng_params["rxnmech"],
                    observables=eng_params["observables"],
                )

        env = DummyVecEnv([lambda: eng])
        agent_params = params.input["agent"]
        if agent_params["agent"] == "calibrated":
            agent = agents.CalibratedAgent(env)
            agent.learn()
        elif agent_params["agent"] == "exhaustive":
            agent = agents.ExhaustiveAgent(env)
            agent.load(fname, env)
        elif agent_params["agent"] == "ppo":
            agent = PPO2.load(fname, env=env)

        df, total_reward = utilities.evaluate_agent(env, agent)
        utilities.plot_df(env, df, idx=k, name=agent_params["agent"])

    utilities.save_plots("compare.pdf")
