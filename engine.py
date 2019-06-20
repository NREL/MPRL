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
class Engine:
    def __init__(self, T0=298.0, p0=103325.0, nsteps=100, fuel="PRF100"):

        self.T0 = T0
        self.p0 = p0
        self.nsteps = nsteps
        self.fuel = fuel_composition(fuel)
        self.ivc = -100
        self.evo = 100
        self.small_mass = 1.0e-15
        self.observables = ["V", "dVdt", "ca", "p"]
        self.internals = ["p", "Tu", "Tb", "mb"]
        self.action_size = 2

        self.reset()

    def reset(self):

        # Setup fuel and oxidizer
        mname = os.path.join("datafiles", "llnl_gasoline_surrogate_323.xml")
        self.gas1 = ct.Solution(mname)
        stoic_ox = 0.0
        for sp, spv in self.fuel.items():
            stoic_ox += (
                self.gas1.n_atoms(self.gas1.species_index(sp), "C") * spv
                + 0.25 * self.gas1.n_atoms(self.gas1.species_index(sp), "H") * spv
            )
        xfu = 0.21 / stoic_ox
        xox = 0.21
        xbath = 1.0 - xfu - xox
        xinit = {}
        for sp, spv in self.fuel.items():
            xinit[sp] = spv * xfu
        xinit["O2"] = xox
        xinit["N2"] = xbath
        self.xinit = xinit
        self.gas1.TPX = self.T0, self.p0, xinit
        self.gas1.equilibrate("HP", solver="gibbs")
        self.xburnt = self.gas1.X
        Tb_ad = self.gas1.T

        # Setup the engine cycle
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
        cycle.p = cycle.p * 1e3 + 101325.0
        cycle.V = cycle.V * 1e-3
        cycle.dVdt = cycle.dVdt * 1e-3 / (0.1 / tscale)
        cycle["t"] = (cycle.ca + 360) / tscale
        cycle = cycle[(cycle.ca > self.ivc) & (cycle.ca < self.evo)]
        self.exact = cycle[["p", "ca", "t", "V"]].copy()

        # interpolate the cycle
        interp = np.linspace(cycle.ca.iloc[0], cycle.ca.iloc[-1], self.nsteps)
        cycle = utilities.interpolate_df(interp, "ca", cycle)

        # Initialize the engine state
        self.states = pd.DataFrame(
            0,
            index=np.arange(len(cycle.index)),
            columns=["V", "dVdt", "ca", "t", "p", "Tu", "Tb", "mb"],
        )
        self.states.V = cycle.V.copy()
        self.states.dVdt = cycle.dVdt.copy()
        self.states["dV"] = self.states.dVdt * (0.1 / tscale)
        self.states.ca = cycle.ca.copy()
        self.states.t = cycle.t.copy()
        self.states.loc[0, self.internals] = [self.p0, self.T0, Tb_ad, 0.0]
        self.current_step = 0

        # Other dataframes
        self.rewards = pd.DataFrame(0, index=self.states.index, columns=["reward"])
        self.rewards.loc[0] = self.states.p.loc[0] * self.states.dV.loc[0]

        return self.get_current_observable_state()

    def step(self, action):
        "Advance the engine to the next state using the action"

        if len(action) != self.action_size:
            sys.exit(f"Error: invalid action size {len(action)} != {self.action_size}")
        mdot, qdot = action

        # Integrate the two zone model between tstart and tend with fixed mdot and qdot
        current_state = self.get_current_state()
        next_state = self.get_next_state()
        integ = ode(
            lambda t, y: self.dfundt_mdot(
                t, y, mdot, next_state.V, next_state.dVdt, Qdot=qdot
            )
        )
        integ.set_initial_value(current_state[self.internals], current_state.t)
        integ.set_integrator("vode", atol=1.0e-8, rtol=1.0e-4)
        integ.integrate(next_state.t)

        self.states.loc[self.current_step + 1, self.internals] = integ.y
        self.current_step += 1

        reward = get_reward(self.get_current_state())
        self.rewards.loc[self.current_step] = reward
        done = self.current_step >= len(self.states) - 1
        return self.get_current_observable_state(), reward, done

    def get_current_state(self):
        return self.states.loc[self.current_step]

    def get_next_state(self):
        return self.states.loc[self.current_step + 1]

    def get_current_observable_state(self):
        return self.states.loc[self.current_step, self.observables]

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
