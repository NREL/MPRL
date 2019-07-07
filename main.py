# ========================================================================
#
# Imports
#
# ========================================================================
import argparse
import numpy as np
import time
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.ddpg.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
    AdaptiveParamNoiseSpec,
)
from stable_baselines import DDPG
from stable_baselines import A2C
from stable_baselines.gail import generate_expert_traj
import engine
import agents
import utilities


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

    # Parse arguments
    parser = argparse.ArgumentParser(description="Train and evaluate an agent")
    parser.add_argument(
        "-a",
        "--agent",
        help="Agent to train and evaluate",
        type=str,
        default="calibrated",
        choices=["calibrated", "a2c", "ddpg"],
    )
    parser.add_argument(
        "-s",
        "--steps",
        help="Total number of steps for training",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-n", "--nranks", help="Number of MPI ranks", type=int, default=1
    )
    parser.add_argument
    args = parser.parse_args()

    # Setup
    start = time.time()
    pa2bar = 1e-5
    nsteps = 100
    np.random.seed(45473)

    # Initialize the engine
    T0 = 273.15 + 120
    p0 = 264_647.769_165_039_06
    eng = engine.Engine(T0=T0, p0=p0, nsteps=nsteps)

    # Create the agent and train
    if args.agent == "calibrated":
        env = DummyVecEnv([lambda: eng])
        agent = agents.CalibratedAgent(env)
        agent.learn()
        agent.generate_expert_traj(args.agent)
    elif args.agent == "ddpg":
        eng.symmetrize_actions()
        env = DummyVecEnv([lambda: eng])
        n_actions = env.action_space.shape[-1]
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions)
        )
        agent = DDPG(
            ddpgMlpPolicy,
            env,
            verbose=1,
            param_noise=param_noise,
            action_noise=action_noise,
        )
        agent.learn(total_timesteps=args.steps)
    elif args.agent == "a2c":
        env = SubprocVecEnv([lambda: eng for i in range(args.nranks)])
        agent = A2C(MlpPolicy, env, verbose=1)
        agent.learn(total_timesteps=args.steps)

    # Save the agent
    agent.save(args.agent)

    # # Evaluate the agent
    fname = args.agent + ".csv"
    df, total_reward = utilities.evaluate_agent(DummyVecEnv([lambda: eng]), agent)
    df.to_csv(fname, index=False)

    # Plots
    plt.figure("mdot")
    plt.plot(df.ca, df.mdot, color=cmap[0], lw=2)

    plt.figure("p")
    plt.plot(df.ca, df.p * pa2bar, color=cmap[0], lw=2)
    plt.plot(eng.exact.ca, eng.exact.p * pa2bar, color=cmap[-1], lw=1)

    plt.figure("p_v")
    plt.plot(df.V, df.p * pa2bar, color=cmap[0], lw=2)
    plt.plot(eng.exact.V, eng.exact.p * pa2bar, color=cmap[-1], lw=1)

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
    plt.plot(df.ca.values.flatten(), np.cumsum(df.rewards), color=cmap[0], lw=2)

    # Save Plots
    fname = f"{args.agent}_time_history.pdf"
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

    # output timer
    end = time.time() - start
    print(f"Elapsed time {timedelta(seconds=end)} (or {end} seconds)")
