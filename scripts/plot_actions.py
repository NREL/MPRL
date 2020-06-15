# ========================================================================
#
# Imports
#
# ========================================================================
import os
import sys
import pickle
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import mprl.inputs as inputs
import mprl.utilities as utilities


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot agent actions")
    parser.add_argument(
        "-a", "--agents", help="Agents to plot", type=str, required=True, nargs="+"
    )
    parser.add_argument(
        "-n", "--nepisodes", help="Number of episodes to plot", type=int, required=True
    )
    args = parser.parse_args()

    # Loop over the folders
    extent = [0, args.nepisodes, -100, 100]
    for k, fdir in enumerate(args.agents):
        run_args = pickle.load(open(os.path.join(fdir, "args.pkl"), "rb"))
        params = inputs.Input()
        params.from_toml(os.path.join(fdir, os.path.basename(run_args.fname)))

        fname = os.path.join(fdir, "actions.npz")
        utilities.plot_actions(
            fname,
            cnt=k,
            nagents=len(args.agents),
            extent=extent,
            frac=params.inputs["agent"]["number_episodes"].value // args.nepisodes,
        )

    utilities.save_action_plots("compare_actions.pdf")
