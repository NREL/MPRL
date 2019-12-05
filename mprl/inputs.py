import sys
import toml
import copy


# ========================================================================
#
# Classes
#
# ========================================================================
class Input:
    def __init__(self):
        self.defaults = {
            "agent": {
                "agent": "ppo",
                "number_episodes": 100,
                "update_nepisodes": 20,
                "nranks": 1,
                "use_pretrained": None,
            },
            "engine": {
                "engine": "twozone-engine",
                "fuel": "dodecane",
                "rxnmech": "dodecane_lu_nox.cti",
                "observables": ["ca", "p", "T", "success_ninj", "can_inject"],
                "nsteps": 201,
                "mdot": 0.234,
                "max_minj": 2.6e-5,
                "max_injections": None,
                "injection_delay": 0.0,
                "small_negative_reward": -200.0,
                "use_qdot": False,
                "use_continuous": False,
            },
        }
        self.input = copy.deepcopy(self.defaults)

    def write_toml(self):
        """Write inputs as TOML format"""
        for section in self.input.keys():
            print(f"""[{section}]""")
            for key, value in self.input[section].items():
                if type(value) is str:
                    print(f"""{key} = "{value}" """)
                else:
                    print(f"""{key} = {value}""")

    def print_help(self):
        """Print the defaults and help"""
        helper = {
            "agent": {
                "agent": "Agent to train and evaluate",
                "number_episodes": "Total number of episodes to train over",
                "update_nepisodes": "Number of episodes per agent update",
                "nranks": "Number of MPI ranks",
                "use_pretrained": "Directory containing a pretrained network to use as a starting point",
            },
            "engine": {
                "engine": "Engine",
                "fuel": "Fuel",
                "rxnmech": "Reaction mechanism",
                "observables": "Engine observables",
                "nsteps": "Engine steps in a given episode",
                "mdot": "Injected mass flow rate [kg/s]",
                "max_minj": "Maximum fuel injected mass [kg]",
                "max_injections": "Maximum number of injections allowed",
                "small_negative_reward": "Negative reward for unallowed actions",
                "injection_delay": "Time delay between injections",
                "use_qdot": "Use a Qdot as an action",
                "use_continuous": "Use a continuous action space",
            },
        }

        for section in self.defaults.keys():
            print(f"""[{section}]""")
            for key, value in helper[section].items():
                if type(self.defaults[section][key]) is str:
                    print(f"""{key} = "{self.defaults[section][key]}" # {value}""")
                else:
                    print(f"""{key} = {self.defaults[section][key]} # {value}""")

    def check(self):
        """Check the inputs"""

        choices = {
            "agent": {"agent": ["calibrated", "exhaustive", "ppo"]},
            "engine": {
                "engine": ["twozone-engine", "reactor-engine", "EQ-engine"],
                "fuel": ["dodecane", "PRF100", "PRF85"],
                "rxnmech": [
                    "dodecane_lu_nox.cti",
                    "dodecane_mars.cti",
                    "dodecane_lu.cti",
                    "llnl_gasoline_surrogate_323.xml",
                ],
            },
        }
        types = {
            "agent": {
                "agent": str,
                "number_episodes": int,
                "update_nepisodes": int,
                "nranks": int,
            },
            "engine": {
                "engine": str,
                "fuel": str,
                "rxnmech": str,
                "observables": list,
                "mdot": float,
                "max_minj": float,
                "injection_delay": float,
                "small_negative_reward": float,
                "use_qdot": bool,
                "use_continuous": bool,
            },
        }

        for section in types.keys():
            for key, value in types[section].items():
                if type(self.input[section][key]) != value:
                    sys.exit(
                        f"""Invalid type: {self.input[section][key]} not {value}"""
                    )

            for key, value in choices[section].items():
                if not self.input[section][key] in value:
                    sys.exit(
                        f"""Invalid choice: {self.input[section][key]} not in {value}"""
                    )

    def from_toml(self, fname):
        """Read TOML file for inputs"""
        parsed = toml.load(fname)
        for section in self.defaults.keys():
            self.input[section].update(parsed[section])
        self.check()
