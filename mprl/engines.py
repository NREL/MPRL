# ========================================================================
#
# Imports
#
# ========================================================================
import os
import sys
import copy
import cantera as ct
import numpy as np
import pandas as pd
from scipy.integrate import ode
import gym
from gym import spaces
import mprl.utilities as utilities
import mprl.actiontypes as actiontypes
import mprl.reward as rw


# ========================================================================
#
# Functions
#
# ========================================================================
def calibrated_engine_ic():
    T0 = 273.15 + 120
    p0 = 264_647.769_165_039_06
    return T0, p0


# ========================================================================
def initial_temp(p, V, bore, stroke, TDCvol):
    """Get the initial temperature from compression"""
    return p / ct.one_atm * (V / (np.pi / 4.0 * bore ** 2 * stroke + TDCvol)) * 300.0


# ========================================================================
def setup_injection_gas(rxnmech, fuel, pure_fuel=True, phi=1.0):
    """Setup the injection gas"""
    gas = ct.Solution(rxnmech)
    far = 0.0

    if fuel == "PRF100":
        gas.set_equivalence_ratio(
            phi, {"IC8H18": 1.0, "NC7H16": 0.0}, {"O2": 1.0, "N2": 3.76}
        )
        afr = (gas.Y[gas.species_index("O2")] + gas.Y[gas.species_index("N2")]) / (
            gas.Y[gas.species_index("IC8H18")]
        )
        far = 1.0 / afr
    elif fuel == "PRF85":
        gas.set_equivalence_ratio(
            phi, {"IC8H18": 0.85, "NC7H16": 0.15}, {"O2": 1.0, "N2": 3.76}
        )
        afr = (gas.Y[gas.species_index("O2")] + gas.Y[gas.species_index("N2")]) / (
            gas.Y[gas.species_index("IC8H18")] + gas.Y[gas.species_index("NC7H16")]
        )
        far = 1.0 / afr
    elif fuel == "dodecane":
        if pure_fuel:
            gas.X = {"NC12H26": 1.0}
        else:
            gas.set_equivalence_ratio(phi, {"NC12H26": 1.0}, {"O2": 1.0, "N2": 3.76})
            afr = (
                gas.Y[gas.species_index("O2")] + gas.Y[gas.species_index("N2")]
            ) / gas.Y[gas.species_index("NC12H26")]
            far = 1.0 / afr
    else:
        sys.exit(f"Unrecognized fuel {fuel}")

    return gas, far


# ========================================================================
def get_nox(gas, mass):
    try:
        no = gas.mass_fraction_dict()["NO"] * mass
    except KeyError:
        no = 0.0
    try:
        no2 = gas.mass_fraction_dict()["NO2"] * mass
    except KeyError:
        no2 = 0.0

    return no + no2


# ========================================================================
def get_soot(gas, mass):
    try:
        return gas.mass_fraction_dict()["C2H2"] * mass
    except KeyError:
        return 0.0


def get_species(gas, mass, name):
    try:
        return gas.mass_fraction_dict()[name] * mass
    except KeyError:
        return 0.0


# ========================================================================
def get_observables_internals(other, histories, observables):
    valid_observables = other + histories
    if len(list(set(valid_observables) & set(observables))) == 0:
        sys.exit(
            f"Selected observables ({observables}) not in valid observables ({valid_observables})"
        )

    internals = list(set(other) - set(observables))
    return observables, internals


# ========================================================================
#
# Classes
#
# ========================================================================
class Engine(gym.Env):
    """An engine environment for OpenAI gym"""

    def __init__(
        self,
        nsteps=100,
        ivc=-100.0,
        evo=100.0,
        fuel="dodecane",
        rxnmech="dodecane_lu_nox.cti",
        max_pressure=200.0,
        ename="Scorpion.xlsx",
        reward=rw.Reward(),
    ):
        """Initialize Engine

        :param nsteps: number of steps
        :type nsteps: int
        :param ivc: crank angle of intake valve closed
        :type ivc: float
        :param evo: crank angle of exhaust valve open
        :type evo: float
        :param fuel: fuel type
        :type fuel: str
        :param rxnmech: mechanism file
        :type rxnmech: str
        :param max_pressure: maximum pressure allowed in engine (atm)
        :type max_pressure: float
        :param ename: file describing the engine
        :type ename: str
        :param reward: reward
        :type reward: Reward()
        :returns: Engine
        :rtype: Engine()

        """

        super(Engine, self).__init__()

        # Engine parameters
        self.T0, self.p0 = calibrated_engine_ic()
        self.nsteps = nsteps
        self.ivc = ivc
        self.evo = evo
        self.fuel = fuel
        self.rxnmech = rxnmech
        self.small_mass = 1.0e-15
        self.max_burned_mass = 6e-3
        self.max_pressure = max_pressure
        self.ename = ename
        self.reward = reward
        self.returns = {k: 0.0 for k in self.reward.names}
        self.rewards = {k: 0.0 for k in self.reward.names}
        self.nepisode = 0
        self.action = None
        self.state_updater = {}
        self.state_reseter = {}
        self.datadir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "datafiles"
        )

        self.observable_attributes = {
            "ca": {
                "low": self.ivc,
                "high": self.evo,
                "scale": 0.5 * (self.evo - self.ivc),
            },
            "p": {
                "low": 0.0,
                "high": np.finfo(np.float32).max,
                "scale": ct.one_atm * 100,
            },
            "T": {"low": 0.0, "high": np.finfo(np.float32).max, "scale": 2000},
            "attempt_ninj": {"low": 0.0, "high": np.iinfo(np.int32).max, "scale": 1.0},
            "success_ninj": {"low": 0.0, "high": np.iinfo(np.int32).max, "scale": 1.0},
            "can_inject": {"low": 0, "high": 1, "scale": 1},
            "nox": {"low": 0.0, "high": np.finfo(np.float32).max, "scale": 1.0},
            "soot": {"low": 0.0, "high": np.finfo(np.float32).max, "scale": 1.0},
        }
        self.observable_attributes.update(self.reward.get_observable_attributes())

    def __repr__(self):
        return self.describe()

    def __str__(self):
        return f"""An instance of {self.describe()}"""

    def __deepcopy__(self, memo):
        """Deepcopy implementation"""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            try:
                setattr(result, k, copy.deepcopy(v, memo))
            except (NotImplementedError, TypeError):
                if "cantera" in v.__class__.__module__:
                    print(
                        f"WARNING: Deepcopy of object containing Cantera members ({k}: {v.__class__}). These are not picklable so we are skipping these. We will assume they are instantiated somehow."
                    )
                else:
                    sys.exit(f"ERROR: in deepcopy of {self.__class__.__name__}")
                pass
        result.setup_lambdas()
        result.setup_cantera()
        return result

    def __getstate__(self):
        """Copy the object's state from self.__dict__"""
        state = self.__dict__.copy()

        # Remove the unpicklable entries.
        lst = []
        for k, v in state.items():
            if "cantera" in v.__class__.__module__:
                lst.append(k)
        for v in lst + self.lambda_names:
            del state[v]

        return state

    def __setstate__(self, state):
        """Restore instance attributes"""
        self.__dict__.update(state)

        # Repopulate the unpicklable entries
        self.setup_lambdas()
        self.setup_cantera()

    def __eq__(self, other):
        """Test for Engine equality

        It is not perfect but tests the big stuff.
        """
        return (
            self.__class__ == other.__class__
            and self.__dict__.keys() == other.__dict__.keys()
            and all(
                [
                    np.allclose(
                        np.linalg.norm(self.history[k]),
                        np.linalg.norm(other.history[k]),
                    )
                    for k in self.history.keys()
                ]
            )
            and all(
                [
                    np.allclose(self.current_state[k], other.current_state[k])
                    for k in self.current_state.keys()
                ]
            )
        )

    def describe(self):
        return f"""{self.__class__.__name__}(nsteps={self.nsteps}, ivc={self.ivc}, evo={self.evo}, fuel="{self.fuel}", rxnmech="{self.rxnmech}", max_pressure={self.max_pressure}, ename="{self.ename}", reward=rw.{self.reward.describe()})"""

    def define_observable_space(self):
        """Define the observable space"""
        obs_low = np.zeros(len(self.observables))
        obs_high = np.zeros(len(self.observables))
        for k, observable in enumerate(self.observables):
            obs_low[k] = self.observable_attributes[observable]["low"]
            obs_high[k] = self.observable_attributes[observable]["high"]

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

    def scale_observables(self, dic):
        sdic = copy.deepcopy(dic)
        for obs in self.observables:
            sdic[obs] /= self.observable_attributes[obs]["scale"]
        return sdic

    def setup_discrete_injection_actions(self):
        """Setup the discrete injection actions"""

        if self.max_injections is None:
            self.max_injections = np.int(np.rint(self.max_minj / (self.mdot * self.dt)))
            print(f"Maximum number of injections is {self.max_injections}")
        else:
            print("Warning: engine setup is overwriting the default max_injections")

        self.action = actiontypes.DiscreteActionType(
            ["mdot"],
            scales={"mdot": self.mdot},
            limits={"mdot": self.max_injections},
            delays={"mdot": self.injection_delay / self.dt},
        )
        self.action_space = self.action.space

    def setup_history(self):
        """Setup the engine history and save for faster reset"""
        ename = os.path.join(self.datadir, self.ename)

        # Engine details
        details = pd.read_excel(ename, sheet_name=["engine"])
        self.bore = details["engine"].bore.values[0]
        self.stroke = details["engine"].stroke.values[0]
        self.rpm = details["engine"].rpm.values[0]
        self.s2ca = 1.0 / (60.0 / self.rpm / 360.0)
        self.total_time = (
            self.evo - self.ivc
        ) / self.s2ca  # Time take to complete (evo - ivc) rotation in seconds

        # Engine cycle
        self.full_cycle = pd.concat(
            [
                pd.read_excel(
                    ename, sheet_name="Ensemble Average", usecols=["PCYL1 - [kPa]_1"]
                ),
                pd.read_excel(ename, sheet_name="Volume"),
            ],
            axis=1,
        )
        self.full_cycle.rename(
            index=str,
            columns={
                "Crank Angle [ATDC]": "ca",
                "Volume [Liter]": "V",
                "PCYL1 - [kPa]_1": "p",
                "dVolume [Liter]": "dV",
            },
            inplace=True,
        )
        l2m3 = 1e-3
        self.full_cycle.p = self.full_cycle.p * 1e3 + 101_325.0
        self.full_cycle.V = self.full_cycle.V * l2m3
        self.full_cycle["t"] = (self.full_cycle.ca + 360) / self.s2ca
        self.TDCvol = self.full_cycle.V.min()

        cycle = self.full_cycle[
            (self.full_cycle.ca >= self.ivc) & (self.full_cycle.ca <= self.evo)
        ]
        self.exact = cycle[["p", "ca", "t", "V"]].copy()

        # interpolate the cycle
        interp, self.dca = np.linspace(self.ivc, self.evo, self.nsteps, retstep=True)
        cycle = utilities.interpolate_df(interp, "ca", cycle)
        self.dt = self.dca / self.s2ca

        # Initialize the engine history
        history_df = pd.DataFrame(
            0.0, index=np.arange(len(cycle.index)), columns=self.histories
        )
        self.starting_cycle_p = cycle.p[0]
        history_df.V = cycle.V.copy()
        history_df.dV = np.gradient(history_df.V)
        history_df.dVdt = history_df.dV / self.dt
        history_df.ca = cycle.ca.copy()
        history_df.dca = self.dca
        history_df.t = cycle.t.copy()
        self.history = history_df.to_dict(orient="list")
        self.history["index"] = history_df.index

    def set_initial_state(self):
        self.p0 = self.starting_cycle_p
        self.T0 = (
            425.0
            if (self.ename == "Scorpion.xlsx")
            else initial_temp(
                self.p0, self.history["V"][0], self.bore, self.stroke, self.TDCvol
            )
        )

    def reset_state(self):
        """Reset the starting state"""
        self.set_initial_state()

        self.current_state = {
            k: 0.0 for k in self.histories + self.observables + self.internals
        }
        self.current_state["name"] = 0

        self.current_state["p"] = self.p0
        self.current_state["T"] = self.T0
        for k in self.histories:
            self.current_state[k] = self.history[k][0]

        self.reward.reset()
        self.returns = {k: 0.0 for k in self.reward.names}
        self.rewards = {k: 0.0 for k in self.reward.names}
        for key, reseter in self.state_reseter.items():
            if key in self.current_state.keys():
                self.current_state[key] = reseter()

    def update_state(self):
        """Update the state"""
        for key in self.current_state.keys():
            if key != "name":
                self.current_state[key] = self.state_updater[key]()
        self.current_state["name"] += 1

    def termination(self):
        """Evaluate termination criteria"""

        done = False
        if self.current_state["name"] >= len(self.history["V"]) - 1:
            done = True

        # Penalties
        penalty = False
        if self.current_state["p"] > self.max_pressure * ct.one_atm:
            penalty = True
            print(
                f"""Maximum pressure ({self.max_pressure} atm) has been exceeded (p = {self.current_state["p"]})!"""
            )
        if self.action.masked:
            penalty = True

        # Compute rewards
        self.rewards = self.reward.compute(
            self.current_state, self.nsteps, penalty, done
        )
        reward = sum(self.rewards.values())
        self.returns = {k: v + self.rewards[k] for k, v in self.returns.items()}

        return reward, done

    def render(self, mode="human", close=False):
        """Render the environment to the screen"""
        print("Nothing to render")

    def get_info(self):
        """Define an information dict for capturing state before reset"""
        self.info = {
            "current_state": self.current_state,
            "returns": self.returns,
            "rewards": self.rewards,
            "reward_weights": self.reward.weights,
        }
        return self.info


# ========================================================================
class TwoZoneEngine(Engine):
    """A two zone engine environment for OpenAI gym"""

    def __init__(self, *args, **kwargs):
        """Initialize TwoZoneEngine (inherits from Engine)"""
        super(TwoZoneEngine, self).__init__(*args, **kwargs)

        # Engine parameters
        self.ode_state = ["p", "Tu", "Tb", "mb"]
        self.histories = ["V", "dVdt", "dV", "ca", "dca", "t"]

        # Engine setup
        self.setup_lambdas()
        self.setup_cantera()
        self.setup_history()

    def setup_lambdas(self):
        """Setup lambda functions.

        The reason for doing it like this is to enable object pickling
        with the standard pickle library. Basically we skip pickling
        these and then manually add them back in.

        """
        self.lambda_names = ["integ", "state_reseter", "state_updater"]

        self.integ = ode(lambda t, y: self.dfundt_mdot(t, y, 0, 0, 0))

        self.state_reseter = {
            "Tu": lambda: self.T0,
            "Tb": lambda: self.Tb_ad,
            "can_inject": lambda: 1,
        }
        self.state_reseter.update(self.reward.get_state_reseter())

        self.state_updater = {
            "p": lambda: self.integ.y[0],
            "Tu": lambda: self.integ.y[1],
            "Tb": lambda: self.integ.y[2],
            "mb": lambda: self.integ.y[3],
            "T": lambda: self.current_state["T"],
            "m": lambda: self.current_state["m"]
            * self.history["V"][self.current_state["name"] + 1],
            "V": lambda: self.history["V"][self.current_state["name"] + 1],
            "dVdt": lambda: self.history["dVdt"][self.current_state["name"] + 1],
            "dV": lambda: self.history["dV"][self.current_state["name"] + 1],
            "ca": lambda: self.history["ca"][self.current_state["name"] + 1],
            "dca": lambda: self.history["dca"][self.current_state["name"] + 1],
            "t": lambda: self.history["t"][self.current_state["name"] + 1],
            "attempt_ninj": lambda: self.action.attempt_counter["mdot"],
            "success_ninj": lambda: self.action.success_counter["mdot"],
            "can_inject": lambda: 1 if self.action.isallowed()["mdot"] else 0,
        }
        self.state_updater.update(self.reward.get_state_updater())

    def setup_cantera(self):
        """Wrapper function to setup all cantera objects"""
        ct.add_directory(self.datadir)
        self.setup_gas()

    def setup_gas(self):
        """Setup the fuel and save for faster reset"""

        injection_gas, self.far = setup_injection_gas(
            self.rxnmech, self.fuel, pure_fuel=False
        )
        injection_gas.TP = self.T0, self.p0
        self.gas = injection_gas
        self.xinit = injection_gas.X
        injection_gas.equilibrate("HP", solver="gibbs")
        self.xburnt = injection_gas.X
        self.Tb_ad = injection_gas.T

    def reset(self):

        super(TwoZoneEngine, self).reset_state()

        self.gas.TPX = self.T0, self.p0, self.xinit

        self.action.reset()

        obs = self.scale_observables(self.current_state)
        return [obs[k] for k in self.observables]

    def step(self, action):
        """Advance the engine to the next state using the action"""

        action = self.action.preprocess(action)

        # Integrate the model using the action
        step = self.current_state["name"]
        self.integ = ode(
            lambda t, y: self.dfundt_mdot(
                t,
                y,
                action["mdot"],
                self.history["V"][step + 1],
                self.history["dVdt"][step + 1],
                Qdot=action["qdot"] if self.action.use_qdot else 0.0,
            )
        )
        self.integ.set_initial_value(
            [self.current_state[k] for k in self.ode_state], self.current_state["t"]
        )
        self.integ.set_integrator("vode", atol=1.0e-8, rtol=1.0e-4)
        self.integ.integrate(self.history["t"][step + 1])

        # Update state
        self.update_state()

        reward, done = self.termination()

        if done:
            print(f"Finished episode #{self.nepisode}")
            self.nepisode += 1

        obs = self.scale_observables(self.current_state)
        return (
            [obs[k] for k in self.observables],
            reward,
            done,
            self.get_info(),
        )

    def dfundt_mdot(self, t, y, mxdot, V, dVdt, Qdot=0.0):
        """
        ODE defining the state evolution.

        :param t: time
        :param y: state, [p, Tu, Tb, mb]
        :param mxdot: rate of injected fuel mass (later converted to mass burning rate)
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
        self.gas.TPX = Tb, p, self.xburnt
        cv_b = self.gas.cv
        ub = self.gas.u  # internal energy
        Rb = 8314.47215 / self.gas.mean_molecular_weight
        Vb = self.gas.v * mb

        # Compute with cantera unburnt gas properties
        self.gas.TPX = Tu, p, self.xinit
        cv_u = self.gas.cv
        cp_u = self.gas.cp
        uu = self.gas.u
        Ru = 8314.47215 / self.gas.mean_molecular_weight
        vu = self.gas.v

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
        #    mbxdot = 0.0

        # Heat exchange rate between the unburned zone and the cylinder walls
        if mb >= self.small_mass:
            Qudot = 0.0
        else:
            Qudot = Qdot

        # Equation A.13, rate of change in the burned mass (get from mdot of fuel)
        mbxdot = mxdot * (1 + 1 / self.far)
        dmbdt = mbxdot

        # Equation A.26, rate of change of the cylinder pressure
        # There is a typo (missing Vu) in the paper (units wouldn't match)
        dpdt = (
            1.0
            / (invgamma_u * Vu - cv_b * RuovRb / cp_u * Vu + cv_b / Rb * V)
            * (
                -1.0 * (1 + cv_b / Rb) * p * dVdt
                - Qdot
                - ((ub - uu) - cv_b * (Tb - RuovRb * Tu)) * mbxdot
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
                * (
                    dVdt
                    - (Vb / mb - Vu / m_u) * mbxdot
                    + V / p * dpdt
                    - Vu / Tu * dTudt
                )
            )

        self.current_state["T"] = (m_u * Tu + mb * Tb) / (m_u + mb)
        self.current_state["m"] = m_u + mb

        return np.array((dpdt, dTudt, dTbdt, dmbdt))


# ========================================================================
class ContinuousTwoZoneEngine(TwoZoneEngine):
    """A two zone engine environment for OpenAI gym

    Description:
        A two-zone model engine is controlled by injecting burned mass.

    Observation:
        Type: Box(4)
        Name   Observation                                Min         Max
        ca     Engine crank angle                         ivc deg     evo deg
        p      Engine pressure                            0           Inf
        T      Gas temperature                            0           Inf
        V      Engine volume                              0           Inf
        dVdt   Engine volume rate of change              -Inf         Inf

    Available actions:
        Type: Box(2)
        Name  Action                                                  Min        Max
        mdot  injection rate of burned mass                           0          max_mdot
        qdot  (optional) heat transfer rate to the cylinder walls    -max_qdot   max_qdot

    Reward:
        Reward is (p dV) for every step taken, including the termination step

    Starting State:
        Initial engine conditions

    Episode Termination:
        Engine reached evo crank angle
        Engine pressure is more than 80bar
        Total injected burned mass is greater than a specified max mass (6e-4 kg)
    """

    def __init__(self, *args, use_qdot=False, **kwargs):
        """Initialize ContinuousTwoZoneEngine (inherits from TwoZoneEngine)

        :param use_qdot: bool to use Qdot as an action
        :type use_qdot: bool
        :returns: ContinuousTwoZoneEngine
        :rtype: ContinuousTwoZoneEngine()

        """
        super(ContinuousTwoZoneEngine, self).__init__(*args, **kwargs)

        # Engine parameters
        self.use_qdot = use_qdot
        self.observables, self.internals = get_observables_internals(
            ["p", "T", "Tu", "Tb", "mb", "m"], self.histories, ["ca"]
        )

        # Final setup
        action_names = ["mdot", "qdot"] if use_qdot else ["mdot"]
        self.action = actiontypes.ContinuousActionType(action_names)
        self.action_space = self.action.space
        self.define_observable_space()
        self.reset()

    def describe(self):
        return f"""{self.__class__.__name__}(nsteps={self.nsteps}, ivc={self.ivc}, evo={self.evo}, fuel="{self.fuel}", rxnmech="{self.rxnmech}", max_pressure={self.max_pressure}, ename="{self.ename}", reward=rw.{self.reward.describe()}, use_qdot={self.use_qdot})"""


# ========================================================================
class DiscreteTwoZoneEngine(TwoZoneEngine):
    """A two zone engine environment for OpenAI gym

    Description:
        A two-zone model engine is controlled by injecting discrete burned mass.

    Observation:
        Type: Box(4)
        Name           Observation                                Min         Max
        ca             Engine crank angle                         ivc deg     evo deg
        p              Engine pressure                            0           Inf
        T              Gas temperature                            0           Inf
        attempt_ninj   Attempted number of injections             0           Inf
        success_ninj   Successful number of injections            0           Inf
        V              Engine volume                              0           Inf
        dVdt           Engine volume rate of change              -Inf         Inf

    Available actions:
        Type: Discrete
        Name  Action
        mdot  injection rate of burned mass

    Reward:
        Reward is (p dV) for every step taken, including the termination step

    Starting State:
        Initial engine conditions

    Episode Termination:
        Engine reached evo crank angle
        Engine pressure is more than 80bar
        Total injected burned mass is greater than a specified max mass (6e-4 kg)
    """

    def __init__(
        self,
        *args,
        mdot=0.1,
        max_minj=5e-05,
        injection_delay=0,
        observables=["ca", "p", "T", "success_ninj", "can_inject"],
        **kwargs,
    ):
        """Initialize DiscreteTwoZoneEngine (inherits from TwoZoneEngine)

        :param mdot: rate of mass injection (kg/s)
        :type mdot: float
        :param max_minj: maximum mass of injected burned fuel/air mixture (kg) allowed
        :type max_minj: float
        :param injection_delay: time delay between injections (s)
        :type injection_delay: float
        :param observables: observables
        :type observables: list
        :returns: DiscreteTwoZoneEngine
        :rtype: DiscreteTwoZoneEngine

        """
        super(DiscreteTwoZoneEngine, self).__init__(*args, **kwargs)

        # Engine parameters
        self.observables, self.internals = get_observables_internals(
            [
                "attempt_ninj",
                "success_ninj",
                "can_inject",
                "p",
                "T",
                "Tu",
                "Tb",
                "mb",
                "m",
            ],
            self.histories,
            observables + self.reward.get_observables(),
        )
        self.mdot = mdot
        self.max_minj = max_minj
        self.max_injections = None
        self.injection_delay = injection_delay

        # Final setup
        self.setup_discrete_injection_actions()
        self.define_observable_space()
        self.reset()

    def describe(self):
        return f"""{self.__class__.__name__}(nsteps={self.nsteps}, ivc={self.ivc}, evo={self.evo}, fuel="{self.fuel}", rxnmech="{self.rxnmech}", max_pressure={self.max_pressure}, ename="{self.ename}", reward=rw.{self.reward.describe()}, mdot={self.mdot}, max_minj={self.max_minj}, injection_delay={self.injection_delay}, observables={self.observables})"""

    def reset(self):

        super(DiscreteTwoZoneEngine, self).reset()

        obs = self.scale_observables(self.current_state)
        return [obs[k] for k in self.observables]


# ========================================================================
class ReactorEngine(Engine):
    """An engine environment for OpenAI gym

    Description:
        A 0D Cantera Reactor engine that injects a fixed composition of fuel/air mixture

    Observation:
        Type: Box(5)
        Name           Observation                                Min         Max
        ca             Engine crank angle                         ivc deg     evo deg
        p              Engine pressure                            0           Inf
        T              Gas temperature                            0           Inf
        attempt_ninj   Attempted number of injections             0           Inf
        success_ninj   Successful number of injections            0           Inf
        V              Engine volume                              0           Inf
        dVdt           Engine volume rate of change              -Inf         Inf

    Available actions:
        Type: Discrete
        Name       Action                           Min        Max
        injection  inject fuel                      0          1

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
        *args,
        Tinj=300.0,
        mdot=0.1,
        max_minj=5e-5,
        injection_delay=0,
        observables=["ca", "p", "T", "success_ninj", "can_inject"],
        **kwargs,
    ):
        """Initialize ReactorEngine (inherits from Engine)

        :param Tinj: injection temperature (K)
        :type Tinj: float
        :param mdot: rate of mass injection (kg/s)
        :type mdot: float
        :param max_minj: maximum mass of injected burned fuel/air mixture (kg) allowed
        :type max_minj: float
        :param injection_delay: time delay between injections (s)
        :type injection_delay: float
        :param observables: observables
        :type observables: list
        :returns: ReactorEngine
        :rtype: ReactorEngine

        """
        super(ReactorEngine, self).__init__(*args, **kwargs)

        # Engine parameters
        self.Tinj = Tinj
        self.histories = ["V", "dVdt", "dV", "ca", "dca", "t", "piston_velocity"]
        self.observables, self.internals = get_observables_internals(
            [
                "attempt_ninj",
                "success_ninj",
                "can_inject",
                "p",
                "T",
                "phi",
                "m",
                "mb",
                "minj",
                "nox",
                "soot",
            ],
            self.histories,
            observables + self.reward.get_observables(),
        )
        self.mdot = mdot
        self.max_minj = max_minj
        self.max_injections = None
        self.injection_delay = injection_delay

        # Engine setup
        self.setup_lambdas()
        self.setup_history()
        self.set_initial_state()
        self.setup_piston()
        self.setup_cantera()
        self.setup_discrete_injection_actions()
        self.define_observable_space()
        self.reset()

    def describe(self):
        return f"""{self.__class__.__name__}(nsteps={self.nsteps}, ivc={self.ivc}, evo={self.evo}, fuel="{self.fuel}", rxnmech="{self.rxnmech}", max_pressure={self.max_pressure}, ename="{self.ename}", reward=rw.{self.reward.describe()}, Tinj={self.Tinj}, mdot={self.mdot}, max_minj={self.max_minj}, injection_delay={self.injection_delay}, observables={self.observables})"""

    def setup_lambdas(self):
        """Setup lambda functions.

        The reason for doing it like this is to enable object pickling
        with the standard pickle library. Basically we skip pickling
        these and then manually add them back in.

        """
        self.lambda_names = ["state_reseter", "state_updater"]

        self.state_reseter = {"can_inject": lambda: 1}
        self.state_reseter.update(self.reward.get_state_reseter())

        self.state_updater = {
            "p": lambda: self.gas.P,
            "T": lambda: self.gas.T,
            "phi": lambda: self.gas.get_equivalence_ratio(),
            "m": lambda: self.reactor.mass,
            "mb": lambda: 0,
            "minj": lambda: self.action.current["mdot"] * self.dt,
            "nox": lambda: get_nox(self.gas, self.reactor.mass),
            "soot": lambda: get_soot(self.gas, self.reactor.mass),
            "V": lambda: self.history["V"][self.current_state["name"] + 1],
            "dVdt": lambda: self.history["dVdt"][self.current_state["name"] + 1],
            "dV": lambda: self.history["dV"][self.current_state["name"] + 1],
            "ca": lambda: self.history["ca"][self.current_state["name"] + 1],
            "dca": lambda: self.history["dca"][self.current_state["name"] + 1],
            "t": lambda: self.history["t"][self.current_state["name"] + 1],
            "piston_velocity": lambda: self.history["piston_velocity"][
                self.current_state["name"] + 1
            ],
            "attempt_ninj": lambda: self.action.attempt_counter["mdot"],
            "success_ninj": lambda: self.action.success_counter["mdot"],
            "can_inject": lambda: 1 if self.action.isallowed()["mdot"] else 0,
        }
        self.state_updater.update(self.reward.get_state_updater())

    def setup_cantera(self):
        """Wrapper function to setup all cantera objects"""
        ct.add_directory(self.datadir)
        self.setup_reactor()

    def setup_piston(self):
        """Calculates the piston velocity given engine history"""
        cylinder_area = np.pi / 4.0 * self.bore ** 2
        self.history["piston_velocity"] = [
            x / (cylinder_area * self.dt) for x in np.diff(self.history["V"])
        ] + [0]

    def setup_reactor(self):
        self.gas = ct.Solution(self.rxnmech)
        self.xinit = {"O2": 0.21, "N2": 0.79}
        self.gas.TPX = self.T0, self.p0, self.xinit

        self.injection_gas, _ = setup_injection_gas(
            self.rxnmech, self.fuel, pure_fuel=True
        )
        self.injection_gas.TP = self.T0, self.p0
        fuel_res = ct.Reservoir(self.injection_gas)

        # Create the reactor object
        self.reactor = ct.Reactor(self.gas)
        self.rempty = ct.Reactor(self.gas)

        # Set the initial states of the reactor
        self.reactor.chemistry_enabled = True
        self.reactor.volume = self.history["V"][0]

        # Add in a fuel injector
        self.injector = ct.MassFlowController(fuel_res, self.reactor)

        # Add in a wall that moves according to piston velocity
        self.piston = ct.Wall(
            left=self.reactor,
            right=self.rempty,
            A=np.pi / 4.0 * self.bore ** 2,
            U=0.0,
            velocity=self.history["piston_velocity"][0],
        )

        # Create the network object
        self.sim = ct.ReactorNet([self.reactor])

    def advance_to_time(self, time):

        self.sim.set_max_time_step(1e-3)
        self.sim.advance(time)

    def reset(self):

        super(ReactorEngine, self).reset_state()

        self.setup_reactor()
        self.gas.TPX = self.T0, self.p0, self.xinit
        self.reactor.volume = self.history["V"][0]
        self.piston.set_velocity(self.history["piston_velocity"][0])
        self.sim.set_initial_time(self.current_state["t"])
        self.current_state["m"] = self.gas.density_mass * self.history["V"][0]

        self.action.reset()

        obs = self.scale_observables(self.current_state)
        return [obs[k] for k in self.observables]

    def step(self, action):
        "Advance the engine to the next state using the action"

        action = self.action.preprocess(action)

        # Integrate the model using the action
        step = self.current_state["name"]
        self.piston.set_velocity(self.current_state["piston_velocity"])

        self.injector.set_mass_flow_rate(action["mdot"])

        self.advance_to_time(self.history["t"][step + 1])

        self.update_state()

        reward, done = self.termination()

        if done:
            print(f"Finished episode #{self.nepisode}")
            self.nepisode += 1

        obs = self.scale_observables(self.current_state)
        return (
            [obs[k] for k in self.observables],
            reward,
            done,
            self.get_info(),
        )


# ========================================================================
class EquilibrateEngine(Engine):
    def __init__(
        self,
        *args,
        Tinj=300.0,
        mdot=0.1,
        max_minj=5e-5,
        injection_delay=0,
        observables=["ca", "p", "T", "success_ninj", "can_inject"],
        **kwargs,
    ):
        """Initialize EquilibrateEngine (inherits from Engine)

        :param Tinj: injection temperature (K)
        :type Tinj: float
        :param mdot: rate of mass injection (kg/s)
        :type mdot: float
        :param max_minj: maximum mass of injected burned fuel/air mixture (kg) allowed
        :type max_minj: float
        :param injection_delay: time delay between injections (s)
        :type injection_delay: float
        :param observables: observables
        :type observables: list
        :returns: EquilibrateEngine
        :rtype: EquilibrateEngine

        """
        super(EquilibrateEngine, self).__init__(*args, **kwargs)

        # Engine parameters
        self.Tinj = Tinj
        self.histories = ["V", "dVdt", "dV", "ca", "dca", "t", "piston_velocity"]
        self.observables, self.internals = get_observables_internals(
            [
                "attempt_ninj",
                "success_ninj",
                "can_inject",
                "p",
                "T",
                "m",
                "mb",
                "minj",
                "nox",
                "soot",
            ],
            self.histories,
            observables + self.reward.get_observables(),
        )

        self.mdot = mdot
        self.max_minj = max_minj
        self.max_injections = None
        self.injection_delay = injection_delay

        # Engine setup
        self.setup_lambdas()
        self.setup_history()
        self.set_initial_state()
        self.setup_cantera()
        self.setup_discrete_injection_actions()
        self.define_observable_space()
        self.reset()

    def describe(self):
        return f"""{self.__class__.__name__}(nsteps={self.nsteps}, ivc={self.ivc}, evo={self.evo}, fuel="{self.fuel}", rxnmech="{self.rxnmech}", max_pressure={self.max_pressure}, ename="{self.ename}", reward=rw.{self.reward.describe()}, Tinj={self.Tinj}, mdot={self.mdot}, max_minj={self.max_minj}, injection_delay={self.injection_delay}, observables={self.observables})"""

    def setup_lambdas(self):
        """Setup lambda functions.

        The reason for doing it like this is to enable object pickling
        with the standard pickle library. Basically we skip pickling
        these and then manually add them back in.

        """
        self.lambda_names = ["state_reseter", "state_updater"]

        self.state_reseter = {"can_inject": lambda: 1}
        self.state_reseter.update(self.reward.get_state_reseter())

        self.state_updater = {
            "p": lambda: self.gas.P,
            "T": lambda: self.gas.T,
            "m": lambda: self.gas.density_mass
            * self.history["V"][self.current_state["name"] + 1],
            "mb": lambda: 0,
            "minj": lambda: self.action.current["mdot"] * self.dt,
            "nox": lambda: get_nox(
                self.gas,
                self.gas.density_mass
                * self.history["V"][self.current_state["name"] + 1],
            ),
            "soot": lambda: get_soot(
                self.gas,
                self.gas.density_mass
                * self.history["V"][self.current_state["name"] + 1],
            ),
            "V": lambda: self.history["V"][self.current_state["name"] + 1],
            "dVdt": lambda: self.history["dVdt"][self.current_state["name"] + 1],
            "dV": lambda: self.history["dV"][self.current_state["name"] + 1],
            "ca": lambda: self.history["ca"][self.current_state["name"] + 1],
            "dca": lambda: self.history["dca"][self.current_state["name"] + 1],
            "t": lambda: self.history["t"][self.current_state["name"] + 1],
            "piston_velocity": lambda: self.history["piston_velocity"][
                self.current_state["name"] + 1
            ],
            "attempt_ninj": lambda: self.action.attempt_counter["mdot"],
            "success_ninj": lambda: self.action.success_counter["mdot"],
            "can_inject": lambda: 1 if self.action.isallowed()["mdot"] else 0,
        }
        self.state_updater.update(self.reward.get_state_updater())

    def setup_cantera(self):
        """Wrapper function to setup all cantera objects"""
        ct.add_directory(self.datadir)
        self.setup_gas()

    def setup_gas(self):
        self.gas = ct.Solution(self.rxnmech)
        self.xinit = {"O2": 0.21, "N2": 0.79}
        self.gas.TPX = self.T0, self.p0, self.xinit
        self.injection_gas, _ = setup_injection_gas(
            self.rxnmech, self.fuel, pure_fuel=True
        )

    def reset(self):
        super(EquilibrateEngine, self).reset_state()

        self.gas.TPX = self.T0, self.p0, self.xinit
        self.current_state["m"] = self.gas.density_mass * self.history["V"][0]
        self.action.reset()

        obs = self.scale_observables(self.current_state)
        return [obs[k] for k in self.observables]

    def step(self, action):
        """Advance the engine to the next state using the action"""

        action = self.action.preprocess(action)

        # Integrate the model using the action
        step = self.current_state["name"]

        gamma = self.gas.cp / self.gas.cv

        P1 = self.gas.P
        V1 = self.history["V"][step]
        V2 = self.history["V"][step + 1]
        P2 = P1 / ((V2 / V1) ** gamma)
        T2 = P2 * V2 / (self.gas.density_mole * V1 * ct.gas_constant)
        self.gas.TP = T2, P2

        if action["mdot"] > 0:
            minj = action["mdot"] * self.dt
            m0 = self.gas.density_mass * V2
            Tnew = (m0 * self.gas.T + minj * self.Tinj) / (m0 + minj)
            Xnew = (m0 * self.gas.X + minj * self.injection_gas.X) / (m0 + minj)

            self.gas.TDX = Tnew, (m0 + minj) / V2, Xnew
            self.gas.equilibrate("UV", solver="auto", rtol=1e-9)

        self.update_state()

        reward, done = self.termination()

        if done:
            print(f"Finished episode #{self.nepisode}")
            self.nepisode += 1

        obs = self.scale_observables(self.current_state)
        return (
            [obs[k] for k in self.observables],
            reward,
            done,
            self.get_info(),
        )
