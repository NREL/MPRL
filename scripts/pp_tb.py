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
def parse_tb(fname, tag):
    df = pd.DataFrame(columns=[tag, "time", "write_time"])
    try:
        for k, event in enumerate(tft.summary_iterator(efile)):
            for v in event.summary.value:
                if v.tag == tag:
                    df.loc[k, tag] = v.simple_value
                    df.loc[k, "write_time"] = datetime.fromtimestamp(event.wall_time)

        # This isn't perfect. Tensorboard writes out events
        # sporadically so several episodes may be recorded at the same
        # wall time. So instead we use the min/max of the write times
        # to find the total wall time (minus the time it took to reach
        # the first write event), and assume each episode took the
        # same amount of time.
        wall_time = (df.write_time.max() - df.write_time.min()).total_seconds()
        df.time = np.linspace(0, wall_time, len(df))

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
    df = parse_tb(efile, "episode_reward")
    df.to_csv(os.path.join(args.fdir, "data.csv"), index=False)
