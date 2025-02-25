from typing import Union, List, Tuple
import sys
from torch import nn
import numpy as np

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


class ALUNETPlanner(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'ALUNETPlanner',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = True):
        print('suppress_transposexxx')
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def determine_transpose(self):
        print('suppress_transpose123XY', self.suppress_transpose)
        sys.exit()
        if self.suppress_transpose:
            return [0, 1, 2], [0, 1, 2]

        # todo we should use shapes for that as well. Not quite sure how yet
        target_spacing = self.determine_fullres_target_spacing()

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        transpose_forward = [max_spacing_axis] + remaining_axes
        transpose_backward = [np.argwhere(np.array(transpose_forward) == i)[0][0] for i in range(3)]
        return transpose_forward, transpose_backward



if __name__ == '__main__':
    ALUNETPlanner(900, 8).plan_experiment()
    print('suppress_transposexxx')
