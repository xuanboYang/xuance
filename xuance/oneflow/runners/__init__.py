from .runner_basic import RunnerBase
from .runner_drl import RunnerDRL
from .runner_marl import RunnerMARL
from .runner_competition import RunnerCompetition
from .runner_pettingzoo import RunnerPettingzoo
from .runner_magent import RunnerMAgent
from .runner_sc2 import RunnerSC2
from .runner_football import RunnerFootball

REGISTRY_Runner = {
    "DL_toolbox": "OneFlow",
    "DRL": RunnerDRL,
    "MARL": RunnerMARL,
    "RunnerCompetition": RunnerCompetition,
    "RunnerPettingzoo": RunnerPettingzoo,
    "RunnerMAgent": RunnerMAgent,
    "RunnerStarCraft2": RunnerSC2,
    "RunnerFootball": RunnerFootball
}

__all__ = [
    "RunnerBase",
    "RunnerDRL",
    "RunnerMARL",
    "RunnerCompetition",
    "RunnerSC2",
    "RunnerFootball",
    "REGISTRY_Runner"
]
