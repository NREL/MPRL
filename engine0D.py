# ========================================================================
#
# Imports
#
# ========================================================================
import os
import sys
import cantera as ct
import numpy as np
import math
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
        A 0D Cantera Reactor engine that injects a fixed composition of fuel/air mixture

    Observation:
        Type: Box(5)
        Name   Observation                       Min         Max
        V      Engine volume                     0           Inf
        dVdt   Engine volume rate of change     -Inf         Inf
        ca     Engine crank angle                ivc deg     evo deg
        p      Engine pressure                   0           Inf
        n_inj  Number of prior injections        0           max_injections

    Available actions:
        Type:   Box(2)
        Name    Action                           Min        Max
        wait    do not inject fuel               0          1
        inject  inject fuel                      0          1

    Reward:
        Reward is (p dV) for every step taken, including the termination step

    Starting State:
        Initial engine conditions

    Episode Termination:
        Engine reached evo crank angle
        Engine pressure is more than 200bar
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        # T0=1000.0,               # Initial temperature of fuel/air mixture (K)
        T0=300.0,
        Tinj=900.0,             # Injection temperature of fuel/air mixture (K)
        minj=0.005,             # Mass of injected fuel/air mixture (kg)
        p0=103_325.0,           # Atmospheric pressure (Pa)
        # nsteps=100,             # Number of steps
        # Bore=99e-3,             # Bore diameter (m)
        Bore=0.0860000029206276,# Bore (m)
        # Stroke=108e-3,          # Stroke length (m)
        Stroke=0.0860000029206276,
        RPM=1495.17016601562,   # RPM of the engine
        # TDCvol=5.5e-5,          # Volume at Top-Dead-Center (m^3)
        TDCvol=6.09216205775738e-5,
        dt=1e-5,                # Time step for the 0D reactor (s)
        max_injections=1,      # Maximum number of injections allowed
        rxnmech='dodecane_mars.cti',
        rxnpath='/Users/nwimer/Documents/Cantera/mechanisms/dodecane_mars',
        # rxnmech='dodecane_lu.cti',
        # rxnpath='/Users/nwimer/Documents/Cantera/mechanisms/dodecane_lu',

        # use_qdot=False,
        # discrete_action=False,
    ):
        super(Engine, self).__init__()

        # Engine parameters
        self.T0 = T0
        self.Tinj = Tinj
        self.minj = minj
        self.p0 = p0
        # self.nsteps = nsteps
        self.Bore = Bore
        self.Stroke = Stroke
        self.RPM = RPM
        self.TDCvol = TDCvol
        self.dt = dt
        self.max_injections = max_injections
        self.rxnmech = rxnmech
        self.rxnpath = rxnpath
        self.ivc = -100
        self.ivc0 = self.ivc
        self.evo = 100
        self.time = 0.0
        self.n_inj = 0
        self.max_pressure = 2000*ct.one_atm
        self.negative_reward = -1e2
        self.huge_negative_reward = -1e16
        self.small_negative_reward = -0.05
        # self.observables = ["V", "dVdt", "ca", "p", "T", "n_inj", "can_inject"]
        self.observables = ["ca", "p", "T", "n_inj", "can_inject"]
        # self.observables = ["ca", "T", "n_inj", "can_inject"]
        self.internals = ["p", "Tu", "Tb", "mb", "T", "n_inj", "can_inject"]
        self.histories = ["p", "V", "dVdt", "dV", "ca", "t", "piston_velocity"]

        # Engine setup
        self.history_setup()
        self.set_initial_state()
        self.engine_setup()
        # self.reset()
        self.define_action_space()
        self.define_observable_space()
        self.reset()

    def define_action_space(self):
        # Define the action space: wait, inject

        # self.actions = ["injection"]
        self.actions = ["mdot"]
        self.action_size = len(self.actions)

        # self.action_counter = {"injection": 0}
        # self.action_limit = {"injection": self.max_injections}
        self.action_counter = {"mdot": 0}
        self.action_limit = {"mdot": self.max_injections}

        self.action_space = spaces.Discrete(2)


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
        return action

    def count_actions(self, action):
        """Keep a running counter of discrete actions"""
        for key in action:
            self.action_counter[key] += action[key]
            # print(self.action_counter[key])

    def mask_action(self, action):
        """Mask actions if they exceed a limit"""
        # for key in action:
            # if (self.action_counter[key] > self.action_limit[key]):
                # self.action_counter[key] -= action[key]
                # action[key] = 0
        if (self.n_inj >= self.max_injections):
            # action["injection"] = 0
            action["mdot"] = 0

        return action

    def preprocess_action(self, action):
        """Preprocess the actions for use by engine"""
        action = self.parse_action(action)
        self.count_actions(action)
        # action = self.mask_action(action)
        return action


    def define_observable_space(self):
        # ["ca", "p", "T", "n_inj", "can_inject"]
        # obs_low = np.array([self.ivc0, 0.0, 0.0, 0, False])
        obs_low = np.array([-100.0, 0.0, 0.0, 0, False])
        obs_high = np.array(
            [
                # self.evo,
                100.0,
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
        cycle = self.full_cycle[(self.full_cycle.ca >= self.ivc) & (self.full_cycle.ca <= self.evo)]
        self.exact = cycle[["p", "ca", "t", "V"]].copy()

        # interpolate the cycle
        self.tend = (self.evo - self.ivc) * (60.0/self.RPM/360.0) # Time take to complete (evo - ivc) rotation in seconds
        self.nsteps = math.ceil(self.tend/self.dt)
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
        self.current_state["T"] = self.T0
        self.current_state[self.histories] = self.history.loc[0, self.histories]
        # ["p", "Tu", "Tb", "mb"]
        # ["p", "Tu", "Tb", "mb", "T", "n_inj", "can_inject"]
        self.current_state[self.internals] = [self.p0, 0.0, 0.0, 0.0, self.T0, 0, True]

        self.scale_observables()

        obs = []
        for item in self.current_state[self.observables]:
            obs.append(item)

        return [obs]


    def set_initial_state(self):
        self.p0 = self.history.p[0]
        self.T0 = (self.p0/ct.one_atm) * (self.history.V[0]/(np.pi/4.0*self.Bore**2 * self.Stroke + self.TDCvol)) * 700.0

    def engine_setup(self):
        """Setup the fuel and reactor"""
        ct.add_directory(self.rxnpath)
        
        # Species and initial conditions for the combustion chamber
        mAirmNC12H26ST = 14.98
        iwtNC12H26 = 1.0/170.33
        iwtAir = 1.0/28.97
        nNC12H26nAirST = (1.0/mAirmNC12H26ST)*(iwtNC12H26/iwtAir)
        # Initial phi0
        phi0 = 0.45
        nNC12H26 = phi0*nNC12H26nAirST/(1.0+phi0*nNC12H26nAirST)
        nAir = 1.0 - nNC12H26
        species_dict_0 = {'NC12H26':nNC12H26, 'O2':0.21*nAir, 'N2':0.79*nAir}
        # Injection phi1
        phi1 = 1.5
        nNC12H26 = phi1*nNC12H26nAirST/(1.0+phi1*nNC12H26nAirST)
        nAir = 1.0 - nNC12H26
        
        self.species_dict_inj = {'NC12H26':nNC12H26, 'O2':0.21*nAir, 'N2':0.79*nAir}

        self.species_dict_air = {'O2':0.21, 'N2':0.79}

        # Define the chamber volume and piston velocity as a function of time (crank angle)
        # Also, overwrites self.p0 to be the pressure at the initial crank angle
        # self.piston_history()
        self.piston_setup()

        self.reactor_setup()

        # self.initial_gas = ct.Solution(self.rxnmech)
        # # self.initial_gas.TPX = self.T0, self.p0, species_dict_0
        # self.initial_gas.TPX = self.T0, self.p0, species_dict_air
        # self.initial_xinit = self.initial_gas.X

        # self.injection_gas = ct.Solution(self.rxnmech)
        # self.injection_gas.TPX = self.Tinj, self.p0, species_dict_inj
        # self.injection_xinit = self.injection_gas.X


        # # Create the reactor object
        # self.gas = self.initial_gas
        # self.reactor = ct.Reactor(self.gas)
        # self.rempty  = ct.Reactor(self.gas)

        # # Set the initial states of the reactor
        # self.reactor.chemistry_enabled = True
        # self.reactor.volume = self.history.V[0]
        # # Add in a wall that moves according to piston velocity
        # self.piston = ct.Wall(left=self.reactor, right=self.rempty,
        #                       A=np.pi/4.0*self.Bore**2, U=0.0, velocity=self.history.piston_velocity[0])

        # # Create the network object
        # self.sim = ct.ReactorNet([self.reactor])


        # self.reactor.chemistry_enabled = True
        # self.reactor.volume = self.chamber_volume[0]

        # # Add in a wall that moves according to Wvel
        # self.piston = ct.Wall(left=self.reactor, right=self.rempty, 
        #                       A=np.pi/4.0*self.Bore**2, U=0.0, velocity=self.piston_velocity[0])

        # # Create the network object
        # self.sim = ct.ReactorNet([self.reactor])


    def piston_setup(self):
        """Calculates the piston velocity given engine history"""

        cylinder_area = np.pi/4.0*self.Bore**2
        piston_velocity = self.history.dVdt / cylinder_area

        self.history.piston_velocity = piston_velocity.copy()

    def reactor_setup(self):
        self.initial_gas = ct.Solution(self.rxnmech)
        # self.initial_gas.TPX = self.T0, self.p0, species_dict_0
        self.initial_gas.TPX = self.T0, self.p0, self.species_dict_air
        self.initial_xinit = self.initial_gas.X

        self.injection_gas = ct.Solution(self.rxnmech)
        self.injection_gas.TPX = self.Tinj, self.p0, self.species_dict_inj
        self.injection_xinit = self.injection_gas.X


        # Create the reactor object
        self.gas = self.initial_gas
        self.reactor = ct.Reactor(self.gas)
        self.rempty  = ct.Reactor(self.gas)

        # Set the initial states of the reactor
        self.reactor.chemistry_enabled = True
        self.reactor.volume = self.history.V[0]
        # Add in a wall that moves according to piston velocity
        self.piston = ct.Wall(left=self.reactor, right=self.rempty,
                              A=np.pi/4.0*self.Bore**2, U=0.0, velocity=self.history.piston_velocity[0])

        # Create the network object
        self.sim = ct.ReactorNet([self.reactor])


    def piston_history(self):
        """Defines the history of the volume chamber and piston velocity"""

        # Ratio of specific heats
        gamma = 1.4

        # Calculate crank angle as a function of time
        time    = np.linspace(0, self.tend, self.nsteps)

        self.CA = (self.evo - self.ivc)/self.tend * time + self.ivc

        # Calculate cylinder head position as a function of crank angle
        Cpos = self.Stroke*np.cos(np.deg2rad(self.CA))
        # Chamber volume as a function of time
        self.chamber_volume = self.TDCvol + (self.Stroke - Cpos)*np.pi/4.0*self.Bore**2
        # Cylinder head velocity as a function of time
        self.piston_velocity = self.Stroke*np.sin(np.deg2rad(self.CA)) * np.deg2rad((self.evo-self.ivc)/self.tend)

        # Initial pressure in the chamber (overwrite self.p0)
        BDCVol  = np.pi/4.0*self.Bore**2 * self.Stroke + self.TDCvol    # Volume at Bottom-Dead-Center
        self.p0 = ((BDCVol/self.chamber_volume[0])**gamma)*ct.one_atm   # Pressure at the initial crank angle (self.ivc)


    def reset(self):

        self.time = 0.0
        # self.history_setup()
        self.set_initial_state()
        # self.engine_setup()
        self.reactor_setup()
        self.sim.set_initial_time(self.time)
        self.n_inj = 0
        # self.gas = self.initial_gas
        # print("Reset pressure = ", self.gas.P/ct.one_atm)
        # print("P0 = ", self.p0/ct.one_atm)
        # self.reactor = ct.Reactor(self.gas)
        # self.rempty  = ct.Reactor(self.gas)
        # # Set the initial states of the reactor
        # self.reactor.chemistry_enabled = True
        # self.reactor.volume = self.history.V[0]
        # # Add in a wall that moves according to piston velocity
        # self.piston = ct.Wall(left=self.reactor, right=self.rempty,
        #                       A=np.pi/4.0*self.Bore**2, U=0.0, velocity=self.history.piston_velocity[0])
        # # Create the network object
        # self.sim = ct.ReactorNet([self.reactor])


        # Initialize the starting state
        self.current_state = pd.Series(
            0.0,
            index=list(
                dict.fromkeys(self.histories + self.observables + self.internals)
            ),
            name=0,
        )
        self.current_state[self.histories] = self.history.loc[0, self.histories]
        # ["p", "Tu", "Tb", "mb", "T", "n_inj", "can_inject"]
        self.current_state[self.internals] = [self.p0, 0.0, 0.0, 0.0, self.T0, self.n_inj, self.n_inj < self.max_injections]

        self.scale_observables()

        return self.current_state[self.observables]

    def step(self, action):
        "Advance the engine to the next state using the action"

        self.unscale_observables()

        # print(action)
        # print(self.current_state.n_inj)
        action = self.preprocess_action(action)

        # If we have reached the max number of injections, action must be "wait"
        # if (self.n_inj >= self.max_injections):
            # action = 0

        step = self.current_state.name
        self.piston.set_velocity(self.current_state.piston_velocity)


        # if (self.action_counter["injection"] > self.action_limit["injection"] and action["injection"]):
        if (self.action_counter["mdot"] > self.action_limit["mdot"] and action["mdot"]):
            print("Injection not allowed!")

            masked_action = action
            # masked_action["injection"] = 0
            masked_action["mdot"] = 0

            self.time += self.dt
            self.sim.advance(self.time)

            self.current_state[self.histories] = self.history.loc[step + 1, self.histories]
            self.current_state[self.internals] = [self.gas.P, 0.0, 0.0, 0.0, self.gas.T, self.n_inj, self.n_inj < self.max_injections]
            self.current_state.name += 1

            # reward, done = self.termination(masked_action["injection"])
            reward, done = self.termination(masked_action["mdot"])

            reward = reward + self.small_negative_reward

            self.scale_observables()
            return (
                self.current_state[self.observables],
                get_reward(self.current_state),
                done,
                # action["injection"],
                action["mdot"],
                {"internals": self.current_state[self.internals]},
            )

        else:
            # if (action["injection"] == 1):
            if (action["mdot"] == 1):
                print("Injecting at crankangle = ", self.current_state.ca)

                m0   = self.gas.density_mass * self.reactor.volume
                Tnew = (m0*self.gas.T + self.minj*self.Tinj)/(m0+self.minj)
                Pnew = self.gas.P
                # print(m0)
                
                Xnew = (m0*self.gas.X + self.minj*self.injection_gas.X)/(m0+self.minj)

                self.gas = ct.Solution(self.rxnmech)
                self.gas.TPX = Tnew, Pnew, Xnew
                self.reactor = ct.Reactor(self.gas)
                self.reactor.chemistry_enabled = True
                self.reactor.volume = self.current_state.V
                self.piston = ct.Wall(left=self.reactor, right=self.rempty,
                                  A=np.pi/4.0*self.Bore**2, U=0.0, velocity=self.current_state.piston_velocity)

                self.sim = ct.ReactorNet([self.reactor])
                self.sim.set_initial_time(self.time)
                self.n_inj += 1

            self.time += self.dt
            self.sim.advance(self.time)

            self.current_state[self.histories] = self.history.loc[step + 1, self.histories]
            self.current_state[self.internals] = [self.gas.P, 0.0, 0.0, 0.0, self.gas.T, self.n_inj, self.n_inj < self.max_injections]
            self.current_state.name += 1

            reward, done = self.termination(action)



        # reward, done = self.termination(action)
        self.scale_observables()
        if done:
            return (
                self.current_state[self.observables],
                reward,
                done,
                # action["injection"],
                action["mdot"],
                {"internals": self.current_state[self.internals]},
            )


        # # Advance the 0D reactor
        # step = self.current_state.name

        # # self.piston.set_velocity(self.piston_velocity[i])
        # self.piston.set_velocity(self.current_state.piston_velocity)

        # if (action["injection"] == 1):
        #     # Inject fuel into the cell and average the temperature/species
        #     print("Injecting at crankangle = ", self.current_state.ca)
        #     m0   = self.gas.density_mass * self.reactor.volume
        #     Tnew = (m0*self.gas.T + self.minj*self.Tinj)/(m0+self.minj)
        #     Pnew = self.gas.P
        #     # print(m0)
            
        #     Xnew = (m0*self.gas.X + self.minj*self.injection_gas.X)/(m0+self.minj)

        #     self.gas = ct.Solution(self.rxnmech)
        #     self.gas.TPX = Tnew, Pnew, Xnew
        #     self.reactor = ct.Reactor(self.gas)
        #     self.reactor.chemistry_enabled = True
        #     self.reactor.volume = self.current_state.V
        #     self.piston = ct.Wall(left=self.reactor, right=self.rempty,
        #                       A=np.pi/4.0*self.Bore**2, U=0.0, velocity=self.current_state.piston_velocity)

        #     self.sim = ct.ReactorNet([self.reactor])
        #     self.sim.set_initial_time(self.time)
        #     self.n_inj += 1

        # # print(self.gas.P/ct.one_atm)
        # # print(self.gas.T)
        # # print(self.gas.P/ct.one_atm, self.gas.T)
        # # print(self.gas.P, self.current_state.p)

        # self.time += self.dt
        # self.sim.advance(self.time)


        # # Update the current state
        # self.current_state[self.histories] = self.history.loc[step + 1, self.histories]
        # # ["p", "Tu", "Tb", "mb", "T", "n_inj", "can_inject"]
        # self.current_state[self.internals] = [self.gas.P, 0.0, 0.0, 0.0, self.gas.T, self.n_inj, self.n_inj < self.max_injections]
        # self.current_state.name += 1

        self.scale_observables()
        return (
            self.current_state[self.observables],
            get_reward(self.current_state),
            done,
            # action["injection"],
            action["mdot"],
            {"internals": self.current_state[self.internals]},
        )



    def termination(self, action):
        """Evaluate termination criteria"""

        done = False
        reward = get_reward(self.current_state)


        if (self.current_state.name >= len(self.history) - 1):
            done = True
        elif (self.current_state.p > self.max_pressure):
            print("Maximum pressure (p = ", self.max_pressure, " ) has been exceeded!")
            done = True
            reward = self.negative_reward

        if (done == True):
            for key in self.actions:
                self.action_counter[key] = 0
            self.n_inj = 0

        return reward, done



    def symmetrize_actions(self):
        """Make action space symmetric (e.g. for DDPG)"""
        self.action_space.low = -self.action_space.high

    def render(self, mode="human", close=False):
        """Render the environment to the screen"""
        print("Nothing to render")

   