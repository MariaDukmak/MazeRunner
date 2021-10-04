from typing import Union
from mazerunner_sim import BatchRunner, HiddenState


class CustomBatchRunner(BatchRunner):
    def __init__(self, filename: str):
        super().__init__(filename)

    def update(env, data: Union[HiddenState, None]) -> HiddenState:
        # stuff
        pass

    def finish(env, data: HiddenState) -> dict:
        # stuff
        pass


batch_runner = CustomBatchRunner()
batch_runner.run_batch()
