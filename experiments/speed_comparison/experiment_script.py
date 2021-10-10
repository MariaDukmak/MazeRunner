from typing import Union
from mazerunner_sim import BatchRunner, HiddenState


if __name__ == '__main_':

    batch_runner = CustomBatchRunner('snelheid_test.feather')
    batch_runner.run_batch(envs=[env], policies=policies, batch_size=10)

    print(feather.read_feather('snelheid_test.feather'))
