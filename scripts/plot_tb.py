# ========================================================================
#
# Imports
#
# ========================================================================
import os
import sys
import argparse
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import mprl.utilities as utilities


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot agents")
    parser.add_argument(
        "-a",
        "--agents",
        help="Folders containing agents to plot",
        type=str,
        required=True,
        nargs="+",
    )
    args = parser.parse_args()

    for k, fdir in enumerate(args.agents):

        # Setup
        fname = os.path.join(fdir, "agent")
        run_args = pickle.load(open(os.path.join(fdir, "args.pkl"), "rb"))

        pfx = "PPO2_1"
        utilities.plot_tb(
            os.path.join(fdir, pfx, "data.csv"), idx=k, name=run_args.agent
        )

    utilities.save_tb_plots("compare_training.pdf")
