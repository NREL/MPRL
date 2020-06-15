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


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Post-process TL in this folder")
    parser.add_argument(
        "-f", "--fname", help="File with output", type=str, required=True
    )
    args = parser.parse_args()

    work = []
    nox = []
    with open(args.fname, "r") as f:
        for line in f:
            if "total work" in line:
                work.append(float(line.split()[-1][:-1]))
            elif "EOC NOx" in line:
                nox.append(float(line.split()[-1][:-1]))

    p_work = [(x - work[-1]) / work[-1] * 100 for x in work]
    p_nox = [(x - nox[-1]) / nox[-1] * 100 for x in nox]

    print(f"Percent difference in work: {p_work}")
    print(f"Percent difference in NOx: {p_nox}")
