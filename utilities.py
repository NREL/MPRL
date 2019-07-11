# ========================================================================
#
# Imports
#
# ========================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate as interp


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
# Functions
#
# ========================================================================
def interpolate_df(x, name, fp):
    """Interpolate a dataframe

    :param x: the x-coordinates at which to evaluate the interpolated values
    :type x: array
    :param name: the name of the column to use in the dataframe for the x-coordinate
    :type name: str
    :param fp: the dataframe containing the y-coordinates
    :type fp: DataFrame
    :returns: the interpolated dataframe
    :rtype: DataFrame
    """
    df = pd.DataFrame({name: x})
    for col in fp.columns:
        f = interp.interp1d(fp[name], fp[col], kind="linear", fill_value="extrapolate")
        df[col] = f(df[name])

    return df


# ========================================================================
def evaluate_agent(env, agent):
    """Evaluate an agent in an engine environment.

    :param env: engine environment
    :type env: Environment
    :param agent: agent
    :type agent: Agent
    :returns: dataframe of history, total rewards
    """

    eng = env.envs[0]

    # Save all the history
    df = pd.DataFrame(
        0.0,
        index=eng.history.index,
        columns=list(
            dict.fromkeys(
                list(eng.history.columns)
                + eng.observables
                + eng.internals
                + eng.actions
                + eng.histories
                + ["rewards"]
            )
        ),
    )
    df[eng.histories] = eng.history[eng.histories]
    df.loc[0, ["rewards"]] = [eng.p0 * eng.history.dV.loc[0]]

    # Evaluate actions from the agent in the environment
    obs = env.reset()
    df.loc[0, eng.observables] = obs
    df.loc[0, eng.internals] = eng.current_state[eng.internals]
    for index in eng.history.index[1:]:
        action, _ = agent.predict(obs)
        obs, reward, done, info = env.step(action)

        # save history
        df.loc[index, eng.actions] = action
        df.loc[index, eng.internals] = info[0]["internals"]
        df.loc[index, ["rewards"]] = reward
        if done:
            break
        df.loc[index, eng.observables] = obs

    df = df.loc[:index, :]

    return df, df.rewards.sum()


# ========================================================================
def plot_df(env, df, idx=0):
    """Make some plots of the agent performance"""

    eng = env.envs[0]
    pa2bar = 1e-5

    plt.figure("mdot")
    p = plt.plot(df.ca, df.mdot, color=cmap[idx], lw=2)
    p[0].set_dashes(dashseq[idx])

    plt.figure("p")
    p = plt.plot(df.ca, df.p * pa2bar, color=cmap[idx], lw=2)
    p[0].set_dashes(dashseq[idx])
    plt.plot(eng.exact.ca, eng.exact.p * pa2bar, color=cmap[-1], lw=1)

    plt.figure("p_v")
    p = plt.plot(df.V, df.p * pa2bar, color=cmap[idx], lw=2)
    p[0].set_dashes(dashseq[idx])
    plt.plot(eng.exact.V, eng.exact.p * pa2bar, color=cmap[-1], lw=1)

    plt.figure("Tu")
    p = plt.plot(df.ca, df.Tu, color=cmap[idx], lw=2)
    p[0].set_dashes(dashseq[idx])

    plt.figure("Tb")
    p = plt.plot(df.ca, df.Tb, color=cmap[idx], lw=2)
    p[0].set_dashes(dashseq[idx])

    plt.figure("mb")
    p = plt.plot(df.ca, df.mb, color=cmap[idx], lw=2)
    p[0].set_dashes(dashseq[idx])

    plt.figure("qdot")
    p = plt.plot(df.ca, df.qdot, color=cmap[idx], lw=2)
    p[0].set_dashes(dashseq[idx])

    plt.figure("reward")
    p = plt.plot(df.ca, df.rewards, color=cmap[idx], lw=2)
    p[0].set_dashes(dashseq[idx])

    plt.figure("cumulative_reward")
    p = plt.plot(df.ca.values.flatten(), np.cumsum(df.rewards), color=cmap[idx], lw=2)
    p[0].set_dashes(dashseq[idx])


# ========================================================================
def save_plots(fname):
    """Save Plots"""

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


# ========================================================================
def plot_training(df, fname):
    """Make some plots of the training"""

    idx = 0

    plt.figure("episode_reward")
    p = plt.plot(df.episode, df.episode_reward, color=cmap[idx], lw=2)
    p[0].set_dashes(dashseq[idx])

    plt.figure("episode_step")
    p = plt.plot(df.episode, df.episode_step, color=cmap[idx], lw=2)
    p[0].set_dashes(dashseq[idx])

    plt.figure("step_rewards")
    plt.scatter(
        df.episode_step,
        df.episode_reward,
        c=cmap[idx],
        alpha=0.2,
        s=15,
        marker=markertype[idx],
    )

    with PdfPages(fname) as pdf:
        plt.figure("episode_reward")
        ax = plt.gca()
        plt.xlabel(r"episode", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\Sigma r$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.tight_layout()
        # legend = ax.legend(loc="best")
        pdf.savefig(dpi=300)

        plt.figure("episode_step")
        ax = plt.gca()
        plt.xlabel(r"episode", fontsize=22, fontweight="bold")
        plt.ylabel(r"step", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.tight_layout()
        # legend = ax.legend(loc="best")
        pdf.savefig(dpi=300)

        plt.figure("step_rewards")
        ax = plt.gca()
        plt.xlabel(r"step", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\Sigma r$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.tight_layout()
        # legend = ax.legend(loc="best")
        pdf.savefig(dpi=300)
