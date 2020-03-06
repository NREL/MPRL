# ========================================================================
#
# Imports
#
# ========================================================================
import os
import sys
import argparse
import numpy as np

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
    parser.add_argument(
        "-l", "--labels", help="Labels for plot", type=str, nargs="+", default=None
    )
    parser.add_argument(
        "-n",
        "--nlims",
        help="Limits on episodes to plot",
        type=int,
        nargs="+",
        default=None,
    )
    args = parser.parse_args()

    # Loop over the folders
    for k, fdir in enumerate(args.fdir):
        fname = os.path.join(fdir, "agent")
        name = None if args.labels is None else args.labels[k]
        limit = np.finfo(float).max if args.nlims is None else args.nlims[k]
        utilities.plot_tb(os.path.join(fdir, "data.csv"), idx=k, name=name, limit=limit)

    utilities.save_tb_plots("compare_training.pdf")
