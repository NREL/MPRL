# ========================================================================
#
# Imports
#
# ========================================================================
import os
import glob
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow.compat.v1.train as tft
import tensorflow.python.framework.errors_impl as tfe


# ========================================================================
#
# Functions
#
# ========================================================================
def parse_tb(fname, tags):
    df = pd.DataFrame(columns=list(tags.keys()) + ["time", "write_time", "step"])
    try:
        for k, event in enumerate(tft.summary_iterator(efile)):
            for v in event.summary.value:
                for key, value in tags.items():
                    if v.tag == value:
                        df.loc[k, key] = v.simple_value
                        df.loc[k, "write_time"] = datetime.fromtimestamp(
                            event.wall_time
                        )
                        df.loc[k, "step"] = event.step

        # This isn't perfect. Tensorboard writes out events
        # sporadically so several episodes may be recorded at the same
        # wall time. So instead we use the min/max of the write times
        # to find the total wall time (minus the time it took to reach
        # the first write event), and assume each episode took the
        # same amount of time.
        df.sort_values(by=["step"], inplace=True)
        wall_time = (df.write_time.max() - df.write_time.min()).total_seconds()
        dt = wall_time / df.step.max()
        df.time = (df.step - df.step.min()) * dt

    except tfe.DataLossError:
        pass

    return df


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
    tags = {
        "episode_reward": "episode_reward",
        "loss": "loss/loss",
        "entropy": "loss/entropy_loss",
    }
    df = parse_tb(efile, tags)
    df.to_csv(os.path.join(args.fdir, "data.csv"), index=False)
