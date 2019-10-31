# ========================================================================
#
# Imports
#
# ========================================================================
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot agents")
    parser.add_argument(
        "-f", "--fdir", help="Folder parameter tuning file", type=str, required=True
    )
    args = parser.parse_args()

    # Read tuning file
    fname = os.path.join(args.fdir, "results.csv")
    df = pd.read_csv(fname)
    completed = df[df.Status == "COMPLETED"]
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(
            completed.sort_values(by=["Objective", "generation"], ascending=False).head(
                n=20
            )
        )

    # Plot the trials
    plt.rc("text", usetex=True)
    plt.figure(0)
    ax = plt.gca()
    grouped = df.groupby("Trial-ID")
    n = len(grouped)
    ax.set_prop_cycle("color", [plt.cm.viridis(i) for i in np.linspace(0, 1, n)])
    colors = plt.cm.viridis(np.linspace(0, 1, completed.generation.max()))

    niter = completed.Iteration.max()
    for name, group in grouped:
        ewma = group.Objective.ewm(alpha=0.05).mean()
        plt.plot(
            (group.generation - 1) * niter + group.Iteration,
            ewma,
            color=colors[group.generation.max() - 1],
        )

    fname = "tuning.pdf"
    with PdfPages(fname) as pdf:
        plt.figure(0)
        ax = plt.gca()
        plt.xlabel(r"iteration", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\Sigma r$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.tight_layout()
        pdf.savefig(dpi=300)

    # Get the data
    nestim = 100
    max_depth = 10
    xcols = [
        "cliprange",
        "ent_coef",
        "gamma",
        "lam",
        "learning_rate",
        "max_grad_norm",
        "n_steps",
        "nminibatches",
        "noptepochs",
        "vf_coef",
    ]
    ycols = ["Objective"]
    Xtrain = completed[xcols]
    Ytrain = completed[ycols].values.flatten()

    # Scale data
    scaler = RobustScaler()
    scaler.fit(Xtrain)
    Xtrain = pd.DataFrame(
        scaler.transform(Xtrain), index=Xtrain.index, columns=Xtrain.columns
    )

    # Train the RF model
    RF = RandomForestRegressor(n_estimators=nestim, max_depth=max_depth, n_jobs=1).fit(
        Xtrain, Ytrain
    )
    importance = pd.DataFrame(
        {"feature": xcols, "importance": RF.feature_importances_}
    ).sort_values(by=["importance"])
    print("Trained RandomForest")
    print("  Score", RF.score(Xtrain, Ytrain))
    print(f"  Features\n{importance}")
