# ========================================================================
#
# Imports
#
# ========================================================================
import os
import glob
import argparse
import numpy as np
import pandas as pd
import tensorflow.compat.v1.train as tft
import tensorflow.python.framework.errors_impl as tfe


# ========================================================================
#
# Functions
#
# ========================================================================
def parse_tb(fname, tag):
    lst = []
    try:
        for event in tft.summary_iterator(efile):
            for v in event.summary.value:
                if v.tag == tag:
                    lst.append(v.simple_value)
    except tfe.DataLossError:
        pass

    return pd.DataFrame({"episode": np.arange(len(lst)), tag: lst})


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot agents")
    parser.add_argument(
        "-f", "--fdir", help="Folder containing TB event file", type=str, required=True
    )
    args = parser.parse_args()

    efile = glob.glob(os.path.join(args.fdir, "events.out.tfevents.*"))[0]
    df = parse_tb(efile, "episode_reward")
    df.to_csv(os.path.join(args.fdir, "data.csv"), index=False)
