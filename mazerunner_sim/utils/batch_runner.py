"""Batch runner class."""

from typing import TypeVar, Union, Sequence, Tuple
import abc
from multiprocessing import Pool
from copy import deepcopy
import pyarrow.feather as feather
import pyarrow as pa
import tqdm

from mazerunner_sim.envs.mazerunner_env import MazeRunnerEnv
from mazerunner_sim.policies.base_policy import BasePolicy

HiddenState = TypeVar('HiddenState')


class BatchRunner(metaclass=abc.ABCMeta):
    """Batch runner class."""

    def __init__(self, filename: str):
        """
        Initialize the batch.
        :param filename: Name of the data file.
        """
        self.filename = filename

    @staticmethod
    @abc.abstractmethod
    def update(env: MazeRunnerEnv, data: Union[HiddenState, None]) -> HiddenState:
        """
        Update function per batch.
        :param env: Mazeenvironment.
        :param data: data that have been generated from a simulation
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def finish(env: MazeRunnerEnv, data: HiddenState) -> dict:
        """
        Finish data of the batch after simulation.
        :param env: Mazeenvironment.
        :param data: data that have been generated from a simulation
        """
        pass

    @classmethod
    def _run_single(cls, env_and_policies: Tuple[MazeRunnerEnv, Sequence[BasePolicy]]) -> Union[None, dict]:
        """
        Run the simulation with the given parameters.
        :param env_and_policies: Tuple of the maze env with the policies of the runners.
        """
        env, policies = env_and_policies
        hidden_state = None
        done = False
        observations = env.get_observations()

        while not done:
            # For every agent, decide an action according to the observation
            actions = {runner_id: policies[runner_id].decide_action(observation) for runner_id, observation in observations.items()}

            # Let the actions take place in the environment
            observations, reward, done, info = env.step(actions)

            # Update data
            hidden_state = cls.update(env, hidden_state)

        summary = cls.finish(env, hidden_state)
        return summary

    def run_batch(self, envs: Sequence[MazeRunnerEnv], policies: Sequence[BasePolicy], batch_size: int) -> None:
        """
        Run a batch of simulations and write the results to a feather file.
        The results are not in order because of multiprocessing, faster simulations are more likely to be at the earlier rows.

        """
        simulator_params = [(deepcopy(envs[i % len(envs)]), policies) for i in range(batch_size)]
        with Pool() as pool:
            results = []
            for result in tqdm.tqdm(pool.imap_unordered(self._run_single, simulator_params), total=batch_size):
                results.append(result)
        # save results
        table = pa.table({column: [row[column] for row in results] for column in results[0].keys()})
        feather.write_feather(table, self.filename)
