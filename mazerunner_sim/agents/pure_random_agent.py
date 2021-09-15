"""Example agent that takes random actions."""
from mazerunner_sim.agents import Agent

import numpy as np
import numpy.typing as npt


class PureRandomAgent(Agent):
    """Create pure random agent."""

    def decide_action(self, observation: npt.NDArray) -> int:
        """Decide action for random agent."""
        return np.random.randint(4)
