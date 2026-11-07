"""hotflash_sim: hot flash, KNDy, vagal, EDA, PRV and skin-temperature simulator.

Quick start:
    >>> from hotflash_sim import simulate_night
    >>> run = simulate_night("menopausal_symptomatic", hours=8, seed=1)
    >>> run["signals"], run["beats"], run["labels"]
"""
__version__ = "0.3.0"

from .params import GROUPS, get_params, param_table
from .simulate import simulate_night, simulate_cohort, calibrate, calibrate_joint
from .stream import RealtimeStream, OnlineFreedmanDetector, play
from .export import plan_scenarios, export_scenario, to_output_rate, verify

__all__ = ["__version__", "GROUPS", "get_params", "param_table", "simulate_night",
           "simulate_cohort", "calibrate", "calibrate_joint", "RealtimeStream",
           "OnlineFreedmanDetector", "play", "plan_scenarios", "export_scenario",
           "to_output_rate", "verify"]
