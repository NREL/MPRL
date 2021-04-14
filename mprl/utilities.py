# ========================================================================
#
# Imports
#
# ========================================================================
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from scipy import interpolate as interp
import pickle
import os


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
rcParams.update({"figure.max_open_warning": 0})
adj = [0.18, 0.14, 0.98, 0.95]


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
        "reactor": "ReactorEngine",
        "EQ": "EQEngine",
        "discrete": "DiscreteTwoZone",
    }
    if name in labels:
        return labels[name]
    else:
        return name


# ========================================================================
def get_fields():
    return {
        "mdot": r"$\dot{m}_f~[\mathrm{kg/s}]$",
        "rewards": r"$r$",
        "T": r"$T~[\mathrm{K}]$",
        "phi": r"$\phi$",
        "Tu": r"$T_u~[\mathrm{K}]$",
        "Tb": r"$T_b~[\mathrm{K}]$",
        "m": r"$m~[\mathrm{kg}]$",
        "mb": r"$m_b~[\mathrm{kg}]$",
        "minj": r"$m_f~[\mathrm{kg}]$",
        "qdot": r"$\dot{Q}~[\mathrm{J/s}]$",
        "nox": r"$m_{NO_x}~[\mathrm{kg}]$",
        "soot": r"$m_{C_2 H_2}~[\mathrm{kg}]$",
        "attempt_ninj": r"attempted \# injections",
        "success_ninj": r"successful \# injections",
        "w_work": r"$\omega_{w}$",
        "w_nox": r"$\omega_{NO_x}$",
        "w_soot": r"$\omega_{C_2 H_2}$",
        "w_penalty": r"$\omega_p$",
        "w_work_nox_label": r"$\omega_{w} = 1 - \omega_{NO_x}$",
        "r_work": r"$r_{w}$",
        "r_nox": r"$r_{NO_x}$",
        "r_soot": r"$r_{C_2 H_2}$",
        "r_penalty": r"$r_p$",
        "work": r"$\Sigma_{t=0}^{N} p_t \Delta V_t$",
        "cumulative_rewards": r"$\Sigma_{t=0}^{N} r_t$",
        "cumulative_r_work": r"$\Sigma_{t=0}^{N} r_{w,t}$",
        "cumulative_r_nox": r"$\Sigma_{t=0}^{N} r_{NO_x,t}$",
        "cumulative_r_soot": r"$\Sigma_{t=0}^{N} r_{C_2 H_2,t}$",
        "cumulative_r_penalty": r"$\Sigma_{t=0}^{N} r_{p,t}$",
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
def pickle_figures(fname):
    dct = {}
    for i in plt.get_fignums():
        fig = plt.figure(i)
        dct[fig.get_label()] = fig
    with open(f"{os.path.splitext(fname)[0]}-fig.pkl", "wb") as f:
        pickle.dump(dct, f)


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
        columns=list(
            dict.fromkeys(
                variables + eng.action.actions + ["rewards"] + eng.reward.get_rewards()
            )
        )
    )

    # Evaluate actions from the agent in the environment
    done = False
    cnt = 0
    obs = env.reset()
    df.loc[cnt, variables] = [eng.current_state[k] for k in variables]
    df.loc[cnt, eng.action.actions] = 0
    rwd = list(eng.reward.compute(eng.current_state, eng.nsteps, False, False).values())
    df.loc[cnt, eng.reward.get_rewards()] = rwd
    df.loc[cnt, ["rewards"]] = [sum(rwd)]

    while not done:
        cnt += 1
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        df.loc[cnt, variables] = [info[0]["current_state"][k] for k in variables]
        df.loc[cnt, eng.action.actions] = eng.action.current
        df.loc[cnt, ["rewards"]] = reward
        df.loc[cnt, eng.reward.get_rewards()] = list(info[0]["rewards"].values())
        if df.loc[cnt, "mdot"] > 0:
            print(f"""Injecting at ca = {df.loc[cnt, "ca"]}""")

    for rwd in eng.reward.get_rewards() + ["rewards"]:
        df[f"cumulative_{rwd}"] = np.cumsum(df[rwd])

    df["work"] = np.cumsum(df.p * df.dV)

    if "phi" in df.columns:
        df.phi.clip(lower=0.0, inplace=True)

    return df, df.rewards.sum()


# ========================================================================
def plot_df(env, df, idx=0, name=None, plot_exp=True):
    """Make some plots of the agent performance"""

    eng = env.envs[0]
    pa2bar = 1e-5
    label = get_label(name)

    cidx = np.mod(idx, len(cmap))
    didx = np.mod(idx, len(dashseq))
    midx = np.mod(idx, len(markertype))

    plt.figure("p")
    _, labels = plt.gca().get_legend_handles_labels()
    if "Exp." not in labels and plot_exp:
        pexp = eng.exact.p * pa2bar
        std_error = 0.04
        plt.plot(eng.exact.ca, pexp, color=cmap[-1], lw=1, label="Exp.")
        plt.fill_between(
            eng.exact.ca,
            pexp - pexp * std_error,
            pexp + pexp * std_error,
            facecolor="0.8",
            color="0.8",
        )
    p = plt.plot(df.ca, df.p * pa2bar, color=cmap[cidx], lw=2, label=label)
    p[0].set_dashes(dashseq[didx])

    plt.figure("p_v")
    _, labels = plt.gca().get_legend_handles_labels()
    if "Exp." not in labels and plot_exp:
        plt.plot(eng.exact.V, eng.exact.p * pa2bar, color=cmap[-1], lw=1, label="Exp.")
    p = plt.plot(df.V, df.p * pa2bar, color=cmap[cidx], lw=2, label=label)
    p[0].set_dashes(dashseq[didx])

    for field in get_fields():
        if field in df.columns:
            plt.figure(field)
            p = plt.plot(df.ca, df[field], color=cmap[cidx], lw=2, label=label)
            p[0].set_dashes(dashseq[didx])

    if plt.fignum_exists("phi") and plt.fignum_exists("T"):
        plt.figure("phi_temp")
        p = plt.plot(
            df["T"],
            df["phi"],
            color=cmap[cidx],
            lw=2,
            label=label,
            ms=5,
            marker=markertype[midx],
        )
        p[0].set_dashes(dashseq[didx])

    for field in ["work", "nox"]:
        figname = f"final_{field}"
        if (field in df.columns) and ("w_work" in df.columns):
            plt.figure(figname)
            plt.plot(
                df.w_work.iloc[-1],
                df[field].iloc[-1],
                color=cmap[0],
                lw=2,
                ms=5,
                marker=markertype[0],
                label=label,
            )


# ========================================================================
def save_plots(fname, legends=["p"]):
    """Save plots"""

    if plt.fignum_exists("phi_temp"):
        datadir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "datafiles",
            "NOx_soot_dodecane_lu_nox.npz",
        )
        data = np.load(datadir)

        NOx = data["NOx"]
        phi = data["phi"]
        temp = data["temp"]

    with PdfPages(fname) as pdf:

        plt.figure("p")
        ax = plt.gca()
        plt.xlabel(r"$\theta$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$p~[\mathrm{bar}]$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        if plt.gcf().get_label() in legends:
            legend = ax.legend(loc="best")
        plt.subplots_adjust(left=adj[0], bottom=adj[1], right=adj[2], top=adj[3])
        pdf.savefig(dpi=300)

        plt.figure("p_v")
        ax = plt.gca()
        plt.xlabel(r"$V$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$p~[\mathrm{bar}]$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(-2, 4))
        if plt.gcf().get_label() in legends:
            legend = ax.legend(loc="best")
        plt.subplots_adjust(left=adj[0], bottom=adj[1], right=adj[2], top=adj[3])
        pdf.savefig(dpi=300)

        fields = get_fields()
        for field, label in fields.items():
            if plt.fignum_exists(field):
                plt.figure(field)
                ax = plt.gca()
                plt.xlabel(r"$\theta$", fontsize=22, fontweight="bold")
                plt.ylabel(label, fontsize=22, fontweight="bold")
                plt.setp(ax.get_xmajorticklabels(), fontsize=16)
                plt.setp(ax.get_ymajorticklabels(), fontsize=16)
                if plt.gcf().get_label() in legends:
                    legend = ax.legend(loc="best")
                plt.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
                plt.subplots_adjust(
                    left=adj[0], bottom=adj[1], right=adj[2], top=adj[3]
                )
                pdf.savefig(dpi=300)

        if plt.fignum_exists("phi_temp"):
            np.clip(NOx, 1e-16, None, out=NOx)
            fig = plt.figure("phi_temp")
            CM = plt.imshow(
                NOx / NOx.max(axis=1).max(axis=0),
                extent=[temp.min(), temp.max(), phi.min(), phi.max()],
                cmap="hot",
                origin="lower",
                aspect="auto",
                norm=LogNorm(vmin=1e-3, vmax=1e0),
            )
            # plt.clim(0, 0.1)
            ax = plt.gca()
            if len(fig.axes) == 1:
                cbar = plt.colorbar(CM)
                cbar.set_label(r"Normalized Y(NO$_x$)")
            plt.xlabel(fields["T"], fontsize=22, fontweight="bold")
            plt.ylabel(fields["phi"], fontsize=22, fontweight="bold")
            plt.setp(ax.get_xmajorticklabels(), fontsize=16)
            plt.setp(ax.get_ymajorticklabels(), fontsize=16)
            ax.set_xlim([500, 2500])
            ax.set_ylim([0, 0.5])
            if plt.gcf().get_label() in legends:
                legend = ax.legend(loc="best")
            plt.subplots_adjust(left=adj[0], bottom=adj[1], right=adj[2], top=adj[3])
            pdf.savefig(dpi=300)

        for field in ["work", "nox"]:
            figname = f"final_{field}"
            if plt.fignum_exists(figname):
                plt.figure(figname)
                ax = plt.gca()
                xlst = [line.get_data()[0][0] for line in ax.lines]
                ylst = [line.get_data()[1][0] for line in ax.lines]
                plt.plot(xlst, ylst, color=cmap[0], lw=2)
                plt.xlabel(fields["w_work_nox_label"], fontsize=22, fontweight="bold")
                plt.ylabel(fields[field], fontsize=22, fontweight="bold")
                plt.setp(ax.get_xmajorticklabels(), fontsize=16)
                plt.setp(ax.get_ymajorticklabels(), fontsize=16)
                plt.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
                plt.subplots_adjust(
                    left=adj[0], bottom=adj[1], right=adj[2], top=adj[3]
                )
                pdf.savefig(dpi=300)

    pickle_figures(fname)


# ========================================================================
def plot_training(df, fname):
    """Make some plots of the training"""

    idx = 0

    cidx = np.mod(idx, len(cmap))
    didx = np.mod(idx, len(dashseq))
    midx = np.mod(idx, len(markertype))

    plt.figure("episode_reward")
    p = plt.plot(df.episode, df.episode_reward, color=cmap[cidx], lw=2)
    p[0].set_dashes(dashseq[didx])

    plt.figure("episode_step")
    p = plt.plot(df.episode, df.episode_step, color=cmap[cidx], lw=2)
    p[0].set_dashes(dashseq[didx])

    plt.figure("step_rewards")
    plt.scatter(
        df.episode_step,
        df.episode_reward,
        c=cmap[cidx],
        alpha=0.2,
        s=15,
        marker=markertype[midx],
    )

    with PdfPages(fname) as pdf:
        plt.figure("episode_reward")
        ax = plt.gca()
        plt.xlabel(r"episode", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\Sigma r$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        # legend = ax.legend(loc="best")
        pdf.savefig(dpi=300)

        plt.figure("episode_step")
        ax = plt.gca()
        plt.xlabel(r"episode", fontsize=22, fontweight="bold")
        plt.ylabel(r"step", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        # legend = ax.legend(loc="best")
        pdf.savefig(dpi=300)

        plt.figure("step_rewards")
        ax = plt.gca()
        plt.xlabel(r"step", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\Sigma r$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        # legend = ax.legend(loc="best")
        pdf.savefig(dpi=300)


# ========================================================================
def plot_tb(fname, alpha=0.1, idx=0, name=None, limit=np.finfo(float).max, lines=[]):
    """Make some plots of tensorboard quantities"""

    label = get_label(name)
    df = pd.read_csv(fname)
    df["episode"] = df.step / 100  # 100 steps per episode
    df = df[df.episode <= limit]
    print(f"""Total training time for {fname}: {df.time.max():.2f} s""")

    cidx = np.mod(idx, len(cmap))
    didx = np.mod(idx, len(dashseq))

    subdf = df.dropna(subset=["episode_reward"])
    ewma = subdf["episode_reward"].ewm(alpha=alpha, adjust=False).mean()

    plt.figure("episode_reward")
    p = plt.plot(subdf.episode, subdf.episode_reward, color=cmap[cidx], lw=2, alpha=0.2)
    p[0].set_dashes(dashseq[didx])
    p = plt.plot(subdf.episode, ewma, color=cmap[cidx], lw=2, label=label)
    p[0].set_dashes(dashseq[didx])

    plt.figure("episode_reward_vs_time")
    p = plt.plot(subdf.time, subdf.episode_reward, color=cmap[cidx], lw=2, alpha=0.2)
    p[0].set_dashes(dashseq[didx])
    p = plt.plot(subdf.time, ewma, color=cmap[cidx], lw=2, label=label)
    p[0].set_dashes(dashseq[didx])

    subdf = df.dropna(subset=["loss"])
    plt.figure("loss")
    p = plt.plot(subdf.episode, subdf.loss, color=cmap[cidx], lw=2, alpha=0.2)
    p[0].set_dashes(dashseq[didx])
    ewma = subdf["loss"].ewm(alpha=alpha, adjust=False).mean()
    p = plt.plot(subdf.episode, ewma, color=cmap[cidx], lw=2, label=label)
    p[0].set_dashes(dashseq[didx])

    for line in lines:
        p = plt.plot([line, line], [1e-6, 1e6], lw=1, color=cmap[-1])
        p[0].set_dashes(dashseq[-1])

    if "entropy" in df.columns:
        subdf = df.dropna(subset=["entropy"])
        plt.figure("entropy")
        p = plt.plot(subdf.episode, subdf.entropy, color=cmap[cidx], lw=2, alpha=0.2)
        p[0].set_dashes(dashseq[didx])
        ewma = subdf["entropy"].ewm(alpha=alpha, adjust=False).mean()
        p = plt.plot(subdf.episode, ewma, color=cmap[cidx], lw=2, label=label)
        p[0].set_dashes(dashseq[didx])
        for line in lines:
            p = plt.plot([line, line], [1e-6, 1e6], lw=1, color=cmap[-1])
            p[0].set_dashes(dashseq[-1])


# ========================================================================
def save_tb_plots(fname, legends=["loss"]):
    """Make some plots of tensorboard quantities"""

    with PdfPages(fname) as pdf:
        plt.figure("episode_reward")
        ax = plt.gca()
        plt.xlabel(r"episode", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\Sigma_{t=0}^{N} r_t$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 3))
        if plt.gcf().get_label() in legends:
            legend = ax.legend(loc="best")
        plt.subplots_adjust(left=adj[0], bottom=adj[1], right=adj[2], top=adj[3])
        pdf.savefig(dpi=300)

        plt.figure("episode_reward_vs_time")
        ax = plt.gca()
        plt.xlabel(r"$t~[s]$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\Sigma_{t=0}^{N} r_t$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 3))
        if plt.gcf().get_label() in legends:
            legend = ax.legend(loc="best")
        plt.subplots_adjust(left=adj[0], bottom=adj[1], right=adj[2], top=adj[3])
        pdf.savefig(dpi=300)

        plt.figure("loss")
        ax = plt.gca()
        plt.xlabel(r"episode", fontsize=22, fontweight="bold")
        plt.ylabel(r"$L_t$", fontsize=22, fontweight="bold")
        # ax.set_xticklabels([f"{int(x/1000)}K" for x in ax.get_xticks().tolist()])
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.setp(ax.get_ymajorticklabels(), fontsize=16)
        plt.ylim([1e-3, 1e5])
        plt.yscale("log")
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 3))
        if plt.gcf().get_label() in legends:
            legend = ax.legend(loc="best")
        plt.subplots_adjust(left=adj[0], bottom=adj[1], right=adj[2], top=adj[3])
        pdf.savefig(dpi=300)

        if plt.fignum_exists("entropy"):
            plt.figure("entropy")
            ax = plt.gca()
            plt.xlabel(r"episode", fontsize=22, fontweight="bold")
            plt.ylabel(r"$S[\pi_t](s_t)$", fontsize=22, fontweight="bold")
            plt.setp(ax.get_xmajorticklabels(), fontsize=16)
            plt.setp(ax.get_ymajorticklabels(), fontsize=16)
            plt.ylim([1e-3, 1e0])
            plt.yscale("log")
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 3))
            if plt.gcf().get_label() in legends:
                legend = ax.legend(loc="best")
            plt.subplots_adjust(left=adj[0], bottom=adj[1], right=adj[2], top=adj[3])
            pdf.savefig(dpi=300)

    pickle_figures(fname)


# ========================================================================
def plot_actions(fname, cnt=0, nagents=1, extent=[0, 10000, -100, 100], frac=1):
    """Make some plots of the actions"""

    npzf = np.load(fname)
    actions = npzf["actions"]
    actions = actions[:, : actions.shape[1] // frac]
    ratio = actions.shape[1] // (actions.shape[0] * nagents)
    if not plt.fignum_exists("actions"):
        fig, axs = plt.subplots(
            num="actions", nrows=nagents, ncols=1, sharex=True, figsize=(8 * ratio, 8)
        )
    else:
        fig = plt.figure("actions")
        axs = fig.get_axes()

    axs[cnt].imshow(
        actions,
        extent=extent,
        interpolation="none",
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    axs[cnt].set_ylabel(r"$\theta$", fontsize=22, fontweight="bold")
    plt.setp(axs[cnt].get_ymajorticklabels(), fontsize=16)


# ========================================================================
def save_action_plots(fname):
    """Make some plots of the actions"""

    with PdfPages(fname) as pdf:
        plt.figure("actions")
        ax = plt.gca()
        ax.set_xlabel(r"episode", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=16)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 3))
        plt.tight_layout()
        pdf.savefig(dpi=400)

    pickle_figures(fname)


# ========================================================================
def grouper(iterable, n):
    """Group iterable in chunks of n"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
