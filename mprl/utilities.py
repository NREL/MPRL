# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate as interp
import mprl.engines as engines


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
def get_label(name):
    labels = {
        "calibrated": "Calibrated",
        "exhaustive": "Exhaustive",
        "ddpg": "DDPG",
        "a2c": "A2C",
        "dqn": "DQN",
        "ppo": "PPO2",
    }
    return labels[name]


# ========================================================================
def get_fields():
    return {
        "mdot": r"$\dot{m}~[\mathrm{kg/s}]$",
        "rewards": r"$r$",
        "T": r"$T~[\mathrm{K}]$",
        "Tu": r"$T_u~[\mathrm{K}]$",
        "Tb": r"$T_b~[\mathrm{K}]$",
        "mb": r"$m_b~[\mathrm{kg}]$",
        "qdot": r"$\dot{Q}~[\mathrm{J/s}]$",
        "nox": r"$Y_{NO_x}$",
        "soot": r"$Y_{C_2 H_2}$",
    }


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
    variables = eng.observables + eng.internals + eng.histories
    df = pd.DataFrame(
        columns=list(dict.fromkeys(variables + eng.action.actions + ["rewards"]))
    )

    # Evaluate actions from the agent in the environment
    done = False
    cnt = 0
    obs = env.reset()
    df.loc[cnt, variables] = eng.current_state[variables]
    df.loc[cnt, eng.action.actions] = 0
    df.loc[cnt, ["rewards"]] = [engines.get_reward(eng.current_state)]

    while not done:
        cnt += 1
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        df.loc[cnt, variables] = info[0]["current_state"][variables]
        df.loc[cnt, eng.action.actions] = eng.action.current
        df.loc[cnt, ["rewards"]] = reward

    return df, df.rewards.sum()


# ========================================================================
def plot_df(env, df, idx=0, name=None):
    """Make some plots of the agent performance"""

    eng = env.envs[0]
    pa2bar = 1e-5
    label = get_label(name)

    plt.figure("p")
    _, labels = plt.gca().get_legend_handles_labels()
    if "Exp." not in labels:
        plt.plot(eng.exact.ca, eng.exact.p * pa2bar, color=cmap[-1], lw=1, label="Exp.")
    p = plt.plot(df.ca, df.p * pa2bar, color=cmap[idx], lw=2, label=label)
    p[0].set_dashes(dashseq[idx])

    plt.figure("p_v")
    _, labels = plt.gca().get_legend_handles_labels()
    if "Exp." not in labels:
        plt.plot(eng.exact.V, eng.exact.p * pa2bar, color=cmap[-1], lw=1, label="Exp.")
    p = plt.plot(df.V, df.p * pa2bar, color=cmap[idx], lw=2, label=label)
    p[0].set_dashes(dashseq[idx])

    for field in get_fields():
        if field in df.columns:
            plt.figure(field)
            p = plt.plot(df.ca, df[field], color=cmap[idx], lw=2, label=label)
            p[0].set_dashes(dashseq[idx])

    plt.figure("cumulative_reward")
    p = plt.plot(
        df.ca.values.flatten(),
        np.cumsum(df.rewards),
        color=cmap[idx],
        lw=2,
        label=label,
    )
    p[0].set_dashes(dashseq[idx])


# ========================================================================
def save_plots(fname):
    """Save plots"""

    with PdfPages(fname) as pdf:

        plt.figure("p")
        ax = plt.gca()
        plt.xlabel(r"$\theta$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$p~[\mathrm{bar}]$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.tight_layout()
        legend = ax.legend(loc="best")
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

        for field, label in get_fields().items():
            if plt.fignum_exists(field):
                plt.figure(field)
                ax = plt.gca()
                plt.xlabel(r"$\theta$", fontsize=22, fontweight="bold")
                plt.ylabel(label, fontsize=22, fontweight="bold")
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


# ========================================================================
def plot_tb(fname, alpha=0.1, idx=0, name=None):
    """Make some plots of tensorboard quantities"""

    label = get_label(name)
    df = pd.read_csv(fname)

    plt.figure("episode_reward")
    p = plt.plot(df.episode, df.episode_reward, color=cmap[idx], lw=2, alpha=0.2)
    p[0].set_dashes(dashseq[idx])

    ewma = df["episode_reward"].ewm(alpha=alpha, adjust=False).mean()
    p = plt.plot(df.episode, ewma, color=cmap[idx], lw=2, label=label)
    p[0].set_dashes(dashseq[idx])


# ========================================================================
def save_tb_plots(fname):
    """Make some plots of tensorboard quantities"""

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
