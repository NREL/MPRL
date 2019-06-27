# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import integrate
import pandas as pd

from stable_baselines.common.vec_env import DummyVecEnv

import engine
import agents


# ========================================================================
#
# Some defaults variables
#
# ========================================================================
plt.rc("text", usetex=True)
cmap_med = [
    "#F15A60",
    "#7AC36A",
    "#5A9BD4",
    "#FAA75B",
    "#9E67AB",
    "#CE7058",
    "#D77FB4",
    "#737373",
]
cmap = [
    "#EE2E2F",
    "#008C48",
    "#185AA9",
    "#F47D23",
    "#662C91",
    "#A21D21",
    "#B43894",
    "#010202",
]
dashseq = [
    (None, None),
    [10, 5],
    [10, 4, 3, 4],
    [3, 3],
    [10, 4, 3, 4, 3, 4],
    [3, 3],
    [3, 3],
]
markertype = ["s", "d", "o", "p", "h"]


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Setup
    pa2bar = 1e-5
    nsteps = 100
    np.random.seed(45473)

    # Initialize the engine
    T0 = 273.15 + 120
    p0 = 264647.76916503906
    engine = engine.Engine(T0=T0, p0=p0, nsteps=nsteps)

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: engine])

    # Create the agent
    agent = agents.CalibratedAgent(env)
    agent.learn()

    # Save all the history
    df = pd.DataFrame(
        0.0,
        index=engine.history.index,
        columns=list(
            dict.fromkeys(
                list(engine.history.columns)
                + engine.observables
                + engine.internals
                + engine.actions
                + engine.histories
                + ["rewards"]
            )
        ),
    )
    df[engine.histories] = engine.history[engine.histories]
    df.loc[0, ["rewards"]] = [engine.p0 * engine.history.dV.loc[0]]

    # Evaluate actions from the agent in the environment
    obs = env.reset()
    df.loc[0, engine.observables] = obs
    df.loc[0, engine.internals] = engine.current_state[engine.internals]

    for index in engine.history.index[1:]:
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)

        # save history
        df.loc[index, engine.actions] = action
        df.loc[index, engine.observables] = obs
        df.loc[index, engine.internals] = info[0]["internals"]
        df.loc[index, ["rewards"]] = reward

    # Plots
    plt.figure("mdot")
    plt.plot(df.ca, df.mdot, color=cmap[0], lw=2)

    plt.figure("p")
    plt.plot(df.ca, df.p * pa2bar, color=cmap[0], lw=2)
    plt.plot(engine.exact.ca, engine.exact.p * pa2bar, color=cmap[-1], lw=1)

    plt.figure("p_v")
    plt.plot(df.V, df.p * pa2bar, color=cmap[0], lw=2)
    plt.plot(engine.exact.V, engine.exact.p * pa2bar, color=cmap[-1], lw=1)

    plt.figure("Tu")
    plt.plot(df.ca, df.Tu, color=cmap[0], lw=2)

    plt.figure("Tb")
    plt.plot(df.ca, df.Tb, color=cmap[0], lw=2)

    plt.figure("mb")
    plt.plot(df.ca, df.mb, color=cmap[0], lw=2)

    plt.figure("qdot")
    plt.plot(df.ca, df.qdot, color=cmap[0], lw=2)

    plt.figure("reward")
    plt.plot(df.ca, df.rewards, color=cmap[0], lw=2)

    plt.figure("cumulative_reward")
    plt.plot(
        df.ca.values.flatten(),
        integrate.cumtrapz(
            df.rewards.values.flatten(), df.ca.values.flatten(), initial=0
        ),
        color=cmap[0],
        lw=2,
    )

    # Save Plots
    fname = "time_history.pdf"
    with PdfPages(fname) as pdf:

        plt.figure("mdot")
        ax = plt.gca()
        plt.xlabel(r"$\theta$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\dot{m}~[\mathrm{kg/s}]$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.tight_layout()
        # legend = ax.legend(loc="best")
        pdf.savefig(dpi=300)

        plt.figure("p")
        ax = plt.gca()
        plt.xlabel(r"$\theta$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$p~[\mathrm{bar}]$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.tight_layout()
        # legend = ax.legend(loc="best")
        pdf.savefig(dpi=300)

        plt.figure("p_v")
        ax = plt.gca()
        plt.xlabel(r"$V$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$p~[\mathrm{bar}]$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.tight_layout()
        # legend = ax.legend(loc="best")
        pdf.savefig(dpi=300)

        plt.figure("Tu")
        ax = plt.gca()
        plt.xlabel(r"$\theta$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$T_u~[\mathrm{K}]$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.tight_layout()
        # legend = ax.legend(loc="best")
        pdf.savefig(dpi=300)

        plt.figure("Tb")
        ax = plt.gca()
        plt.xlabel(r"$\theta$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$T_b~[\mathrm{K}]$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.tight_layout()
        # legend = ax.legend(loc="best")
        pdf.savefig(dpi=300)

        plt.figure("mb")
        ax = plt.gca()
        plt.xlabel(r"$\theta$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$m_b~[\mathrm{kg}]$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.tight_layout()
        # legend = ax.legend(loc="best")
        pdf.savefig(dpi=300)

        plt.figure("qdot")
        ax = plt.gca()
        plt.xlabel(r"$\theta$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\dot{Q}~[\mathrm{J/s}]$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.tight_layout()
        # legend = ax.legend(loc="best")
        pdf.savefig(dpi=300)

        plt.figure("reward")
        ax = plt.gca()
        plt.xlabel(r"$\theta$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$r$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.tight_layout()
        # legend = ax.legend(loc="best")
        pdf.savefig(dpi=300)

        plt.figure("cumulative_reward")
        ax = plt.gca()
        plt.xlabel(r"$\theta$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\Sigma r$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.tight_layout()
        # legend = ax.legend(loc="best")
        pdf.savefig(dpi=300)
