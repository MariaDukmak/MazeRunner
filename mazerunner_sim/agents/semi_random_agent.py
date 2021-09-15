"""Semi-random agent, that goes back in time using pathfinding."""
from mazerunner_sim.agents import Agent
from mazerunner_sim.observation_and_action import Observation, Action

from pathfinding.core.grid import Grid


class SemiRandomAgent(Agent):
    """Semi-random agent, that goes back in time using pathfinding."""

    def decide_action(self, observation: Observation) -> Action:
        """Take a random action, regardless of the observation."""
        grid = Grid(matrix=observation.known_maze)
