# ========================================================================
#
# Imports
#
# ========================================================================
import os
import sys
import argparse

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
        "-f",
        "--fdir",
        help="Folders containing parsed TB event file",
        type=str,
        required=True,
        nargs="+",
    )
    args = parser.parse_args()

    # Loop over the folders
    for k, fdir in enumerate(args.fdir):
        fname = os.path.join(fdir, "agent")
        utilities.plot_tb(os.path.join(fdir, "data.csv"), idx=k)

    utilities.save_tb_plots("compare_training.pdf")
