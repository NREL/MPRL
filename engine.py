# ========================================================================
#
# Imports
#
# ========================================================================
import os
import sys
import cantera as ct
import numpy as np
import pandas as pd
from scipy.integrate import ode
import gym
from gym import spaces
import utilities


# ========================================================================
#
# Functions
#
# ========================================================================
def fuel_composition(fuel):
    if fuel == "PRF85":
        return {"IC8H18": 0.85, "NC7H16": 0.15}
    elif fuel == "PRF100":
        return {"IC8H18": 1.0, "NC7H16": 0.0}
    else:
        sys.exit(f"Unrecognized fuel {fuel}")


def get_reward(state):
    return state.p * state.dV


# ========================================================================
#
# Classes
#
# ========================================================================
class Engine(gym.Env):
    """An engine environment for OpenAI gym

    Description:
        A two-zone model engine is controlled by injecting burned mass.

    Observation:
        Type: Box(4)
        Name   Observation                       Min         Max
        V      Engine volume                     0           Inf
        dVdt   Engine volume rate of change     -Inf         Inf
        ca     Engine crank angle                ivc deg     evo deg
        p      Engine pressure                   0           Inf

    Actions:
        Type: Box(2)
        Name  Action                                       Min        Max
        mdot  injection rate of burned mass                0          max_mdot
        qdot  heat transfer rate to the cylinder walls    -max_qdot   max_qdot

    Reward:
        Reward is (p dV) for every step taken, including the termination step

    Starting State:
        Initial engine conditions

    Episode Termination:
        Engine reached evo crank angle
        Engine pressure is more than 80bar
        Total injected burned mass is greater than a specified max mass (6e-4 kg)
        Injection rate is negative (can't remove burned mass) FIXME: make this part of the action space (cant for DDPG because action space must be symmetric)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, T0=298.0, p0=103_325.0, nsteps=100, fuel="PRF100"):
        super(Engine, self).__init__()

        # Engine parameters
        self.T0 = T0
        self.p0 = p0
        self.nsteps = nsteps
        self.fuel = fuel_composition(fuel)
        self.ivc = -100
        self.evo = 100
        self.small_mass = 1.0e-15
        self.max_burned_mass = 6e-4
        self.max_mdot = 0.5
        self.max_qdot = 0.0
        self.max_pressure = 8e6
        self.negative_reward = -20
        self.observables = ["V", "dVdt", "ca", "p"]
        self.internals = ["p", "Tu", "Tb", "mb"]
        self.actions = ["mdot", "qdot"]
        self.histories = ["V", "dVdt", "dV", "ca", "t"]

        # Define the action space: mdot, qdot
        actions_low = np.array([0, -self.max_qdot])
        actions_high = np.array([self.max_mdot, self.max_qdot])
        self.action_size = len(actions_low)
        self.action_space = spaces.Box(
            low=actions_low, high=actions_high, dtype=np.float16
        )

        # Define the observable space
        obs_low = np.array([0.0, -np.finfo(np.float32).max, self.ivc, 0.0])
        obs_high = np.array(
            [
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                self.evo,
                np.finfo(np.float32).max,
            ]
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        self.fuel_setup()
        self.history_setup()
        self.reset()

    def fuel_setup(self):
        """Setup the fuel and save for faster reset"""
        mname = os.path.join("datafiles", "llnl_gasoline_surrogate_323.xml")
        self.initial_gas = ct.Solution(mname)
        stoic_ox = 0.0
        for sp, spv in self.fuel.items():
            stoic_ox += (
                self.initial_gas.n_atoms(self.initial_gas.species_index(sp), "C") * spv
                + 0.25
                * self.initial_gas.n_atoms(self.initial_gas.species_index(sp), "H")
                * spv
            )
        xfu = 0.21 / stoic_ox
        xox = 0.21
        xbath = 1.0 - xfu - xox
        xinit = {}
        for sp, spv in self.fuel.items():
            xinit[sp] = spv * xfu
        xinit["O2"] = xox
        xinit["N2"] = xbath
        self.initial_xinit = xinit
        self.initial_gas.TPX = self.T0, self.p0, xinit
        self.initial_gas.equilibrate("HP", solver="gibbs")
        self.initial_xburnt = self.initial_gas.X
        self.initial_Tb_ad = self.initial_gas.T

    def history_setup(self):
        """Setup the engine history and save for faster reset"""
        cname = os.path.join("datafiles", "Isooctane_MBT_DI_50C_Summ.xlsx")
        tscale = 9000.0
        cycle = pd.concat(
            [
                pd.read_excel(
                    cname, sheet_name="Ensemble Average", usecols=["PCYL1 - [kPa]_1"]
                ),
                pd.read_excel(cname, sheet_name="Volume"),
            ],
            axis=1,
        )
        cycle.rename(
            index=str,
            columns={
                "Crank Angle [ATDC]": "ca",
                "Volume [Liter]": "V",
                "PCYL1 - [kPa]_1": "p",
                "dVolume [Liter]": "dVdt",
            },
            inplace=True,
        )
        cycle.p = cycle.p * 1e3 + 101_325.0
        cycle.V = cycle.V * 1e-3
        cycle.dVdt = cycle.dVdt * 1e-3 / (0.1 / tscale)
        cycle["t"] = (cycle.ca + 360) / tscale
        cycle = cycle[(cycle.ca > self.ivc) & (cycle.ca < self.evo)]
        self.exact = cycle[["p", "ca", "t", "V"]].copy()

        # interpolate the cycle
        interp = np.linspace(cycle.ca.iloc[0], cycle.ca.iloc[-1], self.nsteps)
        cycle = utilities.interpolate_df(interp, "ca", cycle)

        # Initialize the engine history
        self.history = pd.DataFrame(
            0.0, index=np.arange(len(cycle.index)), columns=self.histories
        )
        self.history.V = cycle.V.copy()
        self.history.dVdt = cycle.dVdt.copy()
        self.history.dV = self.history.dVdt * (0.1 / tscale)
        self.history.ca = cycle.ca.copy()
        self.history.t = cycle.t.copy()

    def reset(self):

        # Reset fuel and oxidizer
        self.gas1 = self.initial_gas
        self.xinit = self.initial_xinit
        self.xburnt = self.initial_xburnt
        self.Tb_ad = self.initial_Tb_ad

        # Initialize the starting state
        self.current_state = pd.Series(
            0.0,
            index=list(
                dict.fromkeys(self.histories + self.observables + self.internals)
            ),
            name=0,
        )
        self.current_state[self.histories] = self.history.loc[0, self.histories]
        self.current_state[self.internals] = [self.p0, self.T0, self.Tb_ad, 0.0]

        return self.current_state[self.observables]

    def step(self, action):
        "Advance the engine to the next state using the action"

        if len(action) != self.action_size:
            sys.exit(f"Error: invalid action size {len(action)} != {self.action_size}")
        mdot, qdot = action

        reward, done = self.termination(mdot)
        if done:
            return (
                self.current_state[self.observables],
                reward,
                done,
                {"internals": self.current_state[self.internals]},
            )

        # Other conditions for stopping
        if mdot < 0:
            done = True
            return (
                self.current_state[self.observables],
                -20,
                done,
                {"internals": self.current_state[self.internals]},
            )

        # Integrate the two zone model between tstart and tend with fixed mdot and qdot
        step = self.current_state.name
        integ = ode(
            lambda t, y: self.dfundt_mdot(
                t,
                y,
                mdot,
                self.history.V.loc[step + 1],
                self.history.dVdt.loc[step + 1],
                Qdot=qdot,
            )
        )
        integ.set_initial_value(
            self.current_state[self.internals], self.current_state.t
        )
        integ.set_integrator("vode", atol=1.0e-8, rtol=1.0e-4)
        integ.integrate(self.history.t.loc[step + 1])

        # Update the current state
        self.current_state[self.histories] = self.history.loc[step + 1, self.histories]
        self.current_state[self.internals] = integ.y
        self.current_state.name += 1

        return (
            self.current_state[self.observables],
            get_reward(self.current_state),
            done,
            {"internals": self.current_state[self.internals]},
        )

    def termination(self, mdot):
        """Evaluate termination criteria"""

        done = False
        reward = get_reward(self.current_state)
        if self.current_state.name >= len(self.history) - 1:
            reward = get_reward(self.current_state)
            done = True
        elif self.current_state.p > self.max_pressure:
            reward = self.negative_reward
            done = True
        elif self.current_state.mb > self.max_burned_mass:
            reward = get_reward(self.current_state)
            done = True
        elif mdot < 0.0:
            reward = self.negative_reward
            done = True

        return reward, done

    def render(self, mode="human", close=False):
        """Render the environment to the screen"""
        print("Nothing to render")

    def dfundt_mdot(self, t, y, mxdot, V, dVdt, Qdot=0.0):
        """
        ODE defining the state evolution.

        :param t: time
        :param y: state, [p, Tu, Tb, mb]
        :param mxdot: mass burning rate
        :param V: volume
        :param dVdot: volume rate of change
        :param Qdot: heat exchange rate between the gases and the cylinder walls
        :returns: ODE

        The equations solved here come from:

        @article{VerhelstS09,
        Author = {S. Verhelst and C.G.W. Sheppard},
        Date-Added = {2016-08-26 14:41:32 +0000},
        Date-Modified = {2016-08-26 14:42:44 +0000},
        Doi = {doi:10.1016/j.enconman.2009.01.002},
        Journal = {Energy Conversion and Management},
        Pages = {1326--1335},
        Title = {Multi-zone thermodynamic modelling of spark-ignition engine combustion -- An overview},
        Volume = {50},
        Year = {2009}}

        The equations are A.21, A.24, A.26.

        In addition, we assume that ml_udot (leakage of unburned gas
        from cylinder to crankcase) is 0 and ml_bdot (leakage of
        burned gas from cylinder to crankcase) is 0.

        """

        p, Tu, Tb, mb = y

        # Compute with cantera burnt gas properties
        self.gas1.TPX = Tb, p, self.xburnt
        cv_b = self.gas1.cv
        ub = self.gas1.u  # internal energy
        Rb = 8314.47215 / self.gas1.mean_molecular_weight
        Vb = self.gas1.v * mb

        # Compute with cantera unburnt gas properties
        self.gas1.TPX = Tu, p, self.xinit
        cv_u = self.gas1.cv
        cp_u = self.gas1.cp
        uu = self.gas1.u
        Ru = 8314.47215 / self.gas1.mean_molecular_weight
        vu = self.gas1.v

        invgamma_u = cv_u / cp_u
        RuovRb = Ru / Rb

        # This compute based on unburned gas EOS, mb, p (get Vb,
        # then V-Vb, or mu and then Vu directly)
        Vu = V - Vb
        m_u = Vu / vu

        # Trim mass burning rate if there isn't any unburned gas left
        # if m_u < 1.0e-10:
        #    mxdot = 0.0

        # Heat exchange rate between the unburned zone and the cylinder walls
        if mb >= self.small_mass:
            Qudot = 0.0
        else:
            Qudot = Qdot

        # Equation A.26, rate of change of the cylinder pressure
        # There is a typo (missing Vu) in the paper (units wouldn't match)
        dpdt = (
            1.0
            / (invgamma_u * Vu - cv_b * RuovRb / cp_u * Vu + cv_b / Rb * V)
            * (
                -1.0 * (1 + cv_b / Rb) * p * dVdt
                - Qdot
                - ((ub - uu) - cv_b * (Tb - RuovRb * Tu)) * mxdot
                + (cv_u / cp_u - cv_b / Rb * Ru / cp_u) * Qudot
            )
        )

        # Equation A.21, rate of change of unburned gas temperature
        dTudt = 1.0 / (m_u * cp_u) * (Vu * dpdt - Qudot)

        # Equation A.24
        if mb <= self.small_mass:
            dTbdt = 0.0
        else:
            dTbdt = (
                p
                / (mb * Rb)
                * (dVdt - (Vb / mb - Vu / m_u) * mxdot + V / p * dpdt - Vu / Tu * dTudt)
            )

        # Equation A.13, rate of change in the burned mass
        dmbdt = mxdot

        return np.array((dpdt, dTudt, dTbdt, dmbdt))
