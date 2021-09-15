"""Example agent that takes random actions."""
from mazerunner_sim.agents import Agent
from mazerunner_sim.observation_and_action import RunnerObservation, Action

import numpy as np


class PureRandomAgent(Agent):
    """Create pure random agent."""

    def observation_action(self, observation: RunnerObservation) -> Action:
        """Take a random action, regardless of the observation."""
        return np.random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])
