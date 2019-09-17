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


# ========================================================================
def get_reward(state):
    return state.p * state.dV


# ========================================================================
def calibrated_engine_ic():
    T0 = 273.15 + 120
    p0 = 264_647.769_165_039_06
    return T0, p0


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

    Available actions:
        Type: Box(2)
        Name  Action                                                  Min        Max
        mdot  injection rate of burned mass                           0          max_mdot
        qdot  (optional) heat transfer rate to the cylinder walls    -max_qdot   max_qdot

        Type: Discrete or Multidiscrete with qdot
        Name  Action
        mdot  injection rate of burned mass
        qdot  (optional) heat transfer rate to the cylinder walls

    Reward:
        Reward is (p dV) for every step taken, including the termination step

    Starting State:
        Initial engine conditions

    Episode Termination:
        Engine reached evo crank angle
        Engine pressure is more than 80bar
        Total injected burned mass is greater than a specified max mass (6e-4 kg)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        T0=298.0,
        p0=103_325.0,
        nsteps=100,
        fuel="PRF100",
        use_qdot=False,
        discrete_action=False,
        ivc=-100.0,
        evo=100.0,
        use_random_start=False,
        subtract_baseline=False,
    ):
        super(Engine, self).__init__()

        # Engine parameters
        T0, p0 = calibrated_engine_ic()
        self.T0 = T0
        self.T = T0
        self.p0 = p0
        self.nsteps = nsteps
        self.fuel = fuel_composition(fuel)
        self.ivc = ivc
        self.ivc0 = ivc
        self.evo = evo
        self.small_mass = 1.0e-15
        self.max_burned_mass = 6e-4
        self.max_injections = 1
        self.max_mdot = 0.5
        self.max_qdot = 0.0
        self.max_pressure = 200*ct.one_atm
        self.negative_reward = -self.nsteps
        # self.small_negative_reward = -1
        # self.small_negative_reward = -0.1
        self.small_negative_reward = -0.05
        # self.noinj_negative_reward = -0.5
        self.noinj_negative_reward = 0.0
        self.observables = ["ca", "p", "T", "n_inj", "can_inject"]
        # self.observables = ["ca", "T", "n_inj", "can_inject"]
        self.internals = ["p", "Tu", "Tb", "mb"]
        self.histories = ["V", "dVdt", "dV", "ca", "t"]
        self.use_qdot = use_qdot
        self.discrete_action = discrete_action
        self.use_random_start = use_random_start
        self.subtract_baseline = subtract_baseline
        self.nepisode = 0

        # Engine setup
        self.fuel_setup()
        self.history_setup()
        self.set_initial_state()
        self.define_action_space()
        self.define_observable_space()
        self.reset()
        if (self.subtract_baseline==True):
            self.establish_baseline()
            self.reset()

    def define_action_space(self):
        # Define the action space: mdot, qdot

        self.actions = ["mdot"]
        if self.use_qdot:
            self.actions.append("qdot")
        self.action_size = len(self.actions)
        print(self.discrete_action)
        if self.discrete_action:
            self.scales = {"mdot": 0.3, "qdot": 0.0}
            self.action_counter = {"mdot": 0, "qdot": 0}
            self.action_limit = {"mdot": self.max_injections, "qdot": 1000}
            if self.use_qdot:
                self.action_space = spaces.MultiDiscrete([2, 1])
            else:
                self.action_space = spaces.Discrete(2)
        else:
            if self.use_qdot:
                actions_low = np.array([0, -self.max_qdot])
                actions_high = np.array([self.max_mdot, self.max_qdot])
            else:
                actions_low = np.array([0])
                actions_high = np.array([self.max_mdot])
            self.action_space = spaces.Box(
                low=actions_low, high=actions_high, dtype=np.float16
            )

    def parse_action(self, action):
        """Create dictionary of actions"""
        action = np.array(action).flatten()
        if len(action) != self.action_size:
            sys.exit(f"Error: invalid action size {len(action)} != {self.action_size}")

        dct = {}
        for k, name in enumerate(self.actions):
            dct[name] = action[k]
        return dct

    def scale_action(self, action):
        """Scale discrete actions to physical space"""
        if self.discrete_action:
            for key in action:
                action[key] *= self.scales[key]

        return action

    def count_actions(self, action):
        """Keep a running counter of discrete actions"""
        if self.discrete_action:
            for key in action:
                self.action_counter[key] += action[key]

    def mask_action(self, action):
        """Mask actions if they exceed a limit"""
        if self.discrete_action:
            for key in action:
                if self.action_counter[key] > self.action_limit[key]:
                    action[key] = 0

        return action

    def preprocess_action(self, action):
        """Preprocess the actions for use by engine"""
        action = self.parse_action(action)
        self.count_actions(action)
        # action = self.mask_action(action)
        # action = self.scale_action(action)
        return action

    def define_observable_space(self):
        # ["ca", "p", "T", "n_inj", "can_inject"]
        obs_low = np.array([self.ivc0, 0.0, 0.0, 0, False])
        # obs_low = np.array([self.ivc0, 0.0, 0, False])
        obs_high = np.array(
            [
                self.evo,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                True,
            ]
        )
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

    def scale_observables(self):
        # self.current_state.p = self.current_state.p/ct.one_atm
        self.current_state.p = self.current_state.p*1e-5

    def unscale_observables(self):
        # self.current_state.p = self.current_state.p*ct.one_atm
        self.current_state.p = self.current_state.p*1e5

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
        self.tscale = 9000.0
        self.full_cycle = pd.concat(
            [
                pd.read_excel(
                    cname, sheet_name="Ensemble Average", usecols=["PCYL1 - [kPa]_1"]
                ),
                pd.read_excel(cname, sheet_name="Volume"),
            ],
            axis=1,
        )
        self.full_cycle.rename(
            index=str,
            columns={
                "Crank Angle [ATDC]": "ca",
                "Volume [Liter]": "V",
                "PCYL1 - [kPa]_1": "p",
                "dVolume [Liter]": "dVdt",
            },
            inplace=True,
        )
        self.full_cycle.p = self.full_cycle.p * 1e3 + 101_325.0
        self.full_cycle.V = self.full_cycle.V * 1e-3
        self.full_cycle.dVdt = self.full_cycle.dVdt * 1e-3 / (0.1 / self.tscale)
        self.full_cycle["t"] = (self.full_cycle.ca + 360) / self.tscale
        
        cycle = self.full_cycle[(self.full_cycle.ca >= self.ivc0) & (self.full_cycle.ca <= self.evo)]
        self.exact = cycle[["p", "ca", "t", "V"]].copy()

        # interpolate the cycle
        interp = np.linspace(self.ivc, self.evo, self.nsteps)
        cycle = utilities.interpolate_df(interp, "ca", cycle)

        # Determine the spacing in crank angle
        self.dca = interp[1] - interp[0]

        # Initialize the engine history
        self.history = pd.DataFrame(
            0.0, index=np.arange(len(cycle.index)), columns=self.histories
        )
        self.history.p = cycle.p.copy()
        self.history.V = cycle.V.copy()
        self.history.dVdt = cycle.dVdt.copy()
        self.history.dV = self.history.dVdt * (0.1 / self.tscale)
        self.history.ca = cycle.ca.copy()
        self.history.t = cycle.t.copy()


    def establish_baseline(self):

        # Run a full simulation without any injection to establish baseline case
        cycle = self.full_cycle[(self.full_cycle.ca >= self.ivc) & (self.full_cycle.ca <= self.evo)]
        self.exact = cycle[["p", "ca", "t", "V"]].copy()

        # interpolate the cycle
        interp = np.linspace(self.ivc0, self.evo, self.nsteps)
        cycle = utilities.interpolate_df(interp, "ca", cycle)

        self.baseline = pd.DataFrame(
            index=np.arange(len(cycle.index)), columns=["p", "dV", "ca", "reward"]
        )

        # Integrate the engine and save reward at each crank angle
        action = self.parse_action(0)
        self.baseline.ca = cycle.ca.copy()
        self.baseline.p = cycle.p.copy()
        self.baseline.dV = self.history.dV.copy()
        self.baseline.reward[self.current_state.name] = get_reward(self.current_state)
        # self.baseline.reward[self.current_state.name] = 0.0
        while (self.current_state["ca"] < self.evo):
            integ = self.integrate_ode(action)
            
            # Update the current state
            self.current_state["n_inj"] = self.action_counter["mdot"]
            self.current_state["can_inject"] = self.action_counter["mdot"] < self.action_limit["mdot"] 
            self.current_state["T"] = self.T
            self.current_state[self.histories] = self.history.loc[self.current_state.name + 1, self.histories]
            self.current_state[self.internals] = integ.y
            self.current_state.name += 1

            # Update the baseline state
            self.baseline.p[self.current_state.name] = self.current_state["p"]
            self.baseline.reward[self.current_state.name] = get_reward(self.current_state)




    def random_start(self):

        if (self.nepisode==0):
            return            

        cycle = self.full_cycle[(self.full_cycle.ca >= self.ivc) & (self.full_cycle.ca <= self.evo)]
        self.exact = cycle[["p", "ca", "t", "V"]].copy()

        # Set a random start crank angle
        interp = np.linspace(self.ivc, self.evo, self.nsteps)
        cycle = utilities.interpolate_df(interp, "ca", cycle)

        ica = np.random.randint(0,self.nsteps-1)

        start_crankangle = interp[ica]

        # Integrate the engine to the start crank angle
        action = self.parse_action(0)
        while (self.current_state["ca"] < start_crankangle):
            integ = self.integrate_ode(action)
            self.current_state["can_inject"] = True 
            self.current_state["n_inj"] = 0
            self.current_state["T"] = self.T
            self.current_state[self.histories] = self.history.loc[self.current_state.name + 1, self.histories]
            self.current_state[self.internals] = integ.y
            self.current_state.name += 1

        print("starting at crankangle = ", self.current_state["ca"])


    def set_full_run(self):

        cycle = self.full_cycle[(self.full_cycle.ca >= self.ivc0) & (self.full_cycle.ca <= self.evo)]
        self.exact = cycle[["p", "ca", "t", "V"]].copy()

        # interpolate the cycle
        interp = np.linspace(self.ivc, self.evo, self.nsteps)
        cycle = utilities.interpolate_df(interp, "ca", cycle)

         # Initialize the engine history
        self.history = pd.DataFrame(
            0.0, index=np.arange(len(cycle.index)), columns=self.histories
        )
        self.history.p = cycle.p.copy()
        self.history.V = cycle.V.copy()
        self.history.dVdt = cycle.dVdt.copy()
        self.history.dV = self.history.dVdt * (0.1 / self.tscale)
        self.history.ca = cycle.ca.copy()
        self.history.t = cycle.t.copy()


        self.current_state = pd.Series(
        0.0,
        index=list(
            dict.fromkeys(self.histories + self.observables + self.internals)
        ),
        name=0,
        )
        self.current_state["n_inj"] = 0
        self.current_state["can_inject"] = True
        self.current_state["T"] = self.T
        self.current_state[self.histories] = self.history.loc[0, self.histories]
        # ["p", "Tu", "Tb", "mb"]
        self.current_state[self.internals] = [self.p0, self.T0, self.Tb_ad, 0.0]

        self.scale_observables()

        obs = []
        for item in self.current_state[self.observables]:
            obs.append(item)

        return [obs]


    def set_initial_state(self):
        self.p0 = self.history.p[0]


    def reset(self):

        # Reset fuel and oxidizer
        self.gas1 = self.initial_gas
        self.xinit = self.initial_xinit
        self.xburnt = self.initial_xburnt
        self.Tb_ad = self.initial_Tb_ad

        self.set_initial_state()

        for key in self.actions:
                self.action_counter[key] = 0

        # Initialize the starting state
        self.current_state = pd.Series(
            0.0,
            index=list(
                dict.fromkeys(self.histories + self.observables + self.internals)
            ),
            name=0,
        )
        self.current_state["n_inj"] = 0
        self.current_state["can_inject"] = True
        self.current_state["T"] = self.T
        self.current_state[self.histories] = self.history.loc[0, self.histories]
        # ["p", "Tu", "Tb", "mb"]
        self.current_state[self.internals] = [self.p0, self.T0, self.Tb_ad, 0.0]

        if (self.use_random_start==True):
            self.random_start()

        self.scale_observables()

        return self.current_state[self.observables]


    def integrate_ode(self, action):
        # Integrate the two zone model between tstart and tend with fixed mdot and qdot
        step = self.current_state.name
        integ = ode(
            lambda t, y: self.dfundt_mdot(
                t,
                y,
                action["mdot"]*self.scales["mdot"],
                self.history.V.loc[step + 1],
                self.history.dVdt.loc[step + 1],
                Qdot=action["qdot"] if self.use_qdot else 0.0,
            )
        )
        integ.set_initial_value(
            self.current_state[self.internals], self.current_state.t
        )
        integ.set_integrator("vode", atol=1.0e-8, rtol=1.0e-4)
        integ.integrate(self.history.t.loc[step + 1])


        return integ

    def step(self, action):
        "Advance the engine to the next state using the action"

        self.unscale_observables()

        action = self.preprocess_action(action)

        # reward, done = self.termination(action["mdot"])

        if (self.action_counter["mdot"] > self.action_limit["mdot"] and action["mdot"]==1):
            # We cannot inject, give a small negative reward and return without integrating the engine state
            print("Injection not allowed!")

            masked_action = action
            masked_action["mdot"] = 0
            integ = self.integrate_ode(masked_action)

            self.current_state["can_inject"] = self.action_counter["mdot"] < self.action_limit["mdot"] 
            self.current_state["n_inj"] = self.action_counter["mdot"]
            self.current_state["T"] = self.T
            self.current_state[self.histories] = self.history.loc[self.current_state.name + 1, self.histories]
            self.current_state[self.internals] = integ.y
            self.current_state.name += 1

            reward, done = self.termination(masked_action["mdot"])

            if (self.subtract_baseline==True):
                reward = get_reward(self.current_state) - self.baseline.reward[self.current_state.name]
            else:
                reward = get_reward(self.current_state)
            # reward = reward + self.small_negative_reward*self.action_counter["mdot"]
            reward = reward + self.small_negative_reward

            self.scale_observables()
            return (
                self.current_state[self.observables],
                reward,
                done,
                action["mdot"],
                {"internals": self.current_state[self.internals]},
            )
        else:
            if (action["mdot"] > 0):
                print("Injecting at crankangle = ", self.current_state.ca)

            integ = self.integrate_ode(action)
            # Update the current state
            self.current_state["n_inj"] = self.action_counter["mdot"]
            self.current_state["can_inject"] = self.action_counter["mdot"] < self.action_limit["mdot"] 
            self.current_state["T"] = self.T
            self.current_state[self.histories] = self.history.loc[self.current_state.name + 1, self.histories]
            self.current_state[self.internals] = integ.y
            self.current_state.name += 1

            reward, done = self.termination(action["mdot"])


        if done:
            print('Finished episode #', self.nepisode)
            self.nepisode += 1
            if (self.subtract_baseline==True):
                reward = reward - self.baseline.reward[self.current_state.name]

            self.scale_observables()
            return (
                self.current_state[self.observables],
                reward,
                done,
                action["mdot"],
                {"internals": self.current_state[self.internals]},
            )

        # if (action["mdot"] > 0):
        #     print("Injecting at crankangle = ", self.current_state.ca)
            # reward += self.small_negative_reward

        # integ = self.integrate_ode(action)

        # # Ensure that the integration was successful
        # if not integ.successful():
        #     print(f"Integration failure (return code = {integ.get_return_code()})")
        #     return (
        #         self.current_state[self.observables],
        #         self.negative_reward,
        #         True,
        #         action["mdot"],
        #         {"internals": self.current_state[self.internals]},
        #     )


        # # Update the current state
        # self.current_state["n_inj"] = self.action_counter["mdot"]
        # self.current_state["can_inject"] = self.action_counter["mdot"] < self.action_limit["mdot"] 
        # self.current_state["T"] = self.T
        # self.current_state[self.histories] = self.history.loc[self.current_state.name + 1, self.histories]
        # self.current_state[self.internals] = integ.y
        # self.current_state.name += 1

        # if (self.subtract_baseline==True):
        #     reward = get_reward(self.current_state)-self.baseline.reward[self.current_state.name]
        # else:
        #     reward = get_reward(self.current_state)
        
        # if (self.action_counter["mdot"] == 0):
        #     reward = reward + self.noinj_negative_reward
            # reward = np.minimum(self.noinj_negative_reward, reward)
            
        self.scale_observables()
        return (
            self.current_state[self.observables],
            reward,
            done,
            action["mdot"],
            {"internals": self.current_state[self.internals]},
        )

    def termination(self, mdot):
        """Evaluate termination criteria"""

        done = False
        if (self.subtract_baseline==True):
            reward = get_reward(self.current_state)-self.baseline.reward[self.current_state.name]
        else:
            reward = get_reward(self.current_state)
        if (self.current_state.name >= len(self.history) - 1) or (
            self.current_state.mb > self.max_burned_mass
        ):
            done = True
        elif (mdot < -1e-3) or ((self.current_state.p > self.max_pressure)):
            print("Maximum pressure (p = ", self.max_pressure, " ) has been exceeded!")
            done = True
            reward = self.negative_reward

        if (done == True):
            for key in self.actions:
                self.action_counter[key] = 0

        return reward, done

    def symmetrize_actions(self):
        """Make action space symmetric (e.g. for DDPG)"""
        self.action_space.low = -self.action_space.high

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
        # Vu = np.maximum(Vu,self.small_mass)
        # if Vu < 0:
            # print("Volume is negative!!!")
            # exit()
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

        self.T = (m_u*Tu + mb*Tb)/(m_u + mb)

        return np.array((dpdt, dTudt, dTbdt, dmbdt))
