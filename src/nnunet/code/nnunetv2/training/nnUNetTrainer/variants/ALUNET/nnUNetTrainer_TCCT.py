# import torch
# from typing import Union, Tuple, List
# from batchgenerators.transforms.abstract_transforms import AbstractTransform
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# import numpy as np
# from torch import autocast
# from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss, DC_and_CE_loss_CACS
# from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
# from nnunetv2.utilities.helpers import dummy_context
# from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
# from torch.nn.parallel import DistributedDataParallel as DDP

# import inspect
# import multiprocessing
# import os
# import shutil
# import sys
# import warnings
# from copy import deepcopy
# from datetime import datetime
# from time import time, sleep
# from typing import Union, Tuple, List
# from os.path import join
# import time
# from helper.helper import splitFilePath

# from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
# from nnunetv2.utilities.helpers import empty_cache, dummy_context
# from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
# from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
# from torch import distributed as dist
# from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
# from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper


# from nnunetv2.utilities.label_handling.label_handling import LabelManager
# from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
# from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
# from nnunetv2.inference.predict_from_raw_data import compute_steps_for_sliding_window
# from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
# from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
# from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D

# from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision


# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
# #from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
# from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
# from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
# from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
# from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
# from nnunetv2.inference.sliding_window_prediction import compute_gaussian

import os, sys
import warnings
import torch
import numpy as np
from time import time, sleep
import multiprocessing
from os.path import join
from torch import distributed as dist
from nnunetv2.training.nnUNetTrainer.variants.ALUNET.nnUNetTrainer_ALUNET import nnUNetTrainer_ALUNET

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
#from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
#from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p#
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results

class nnUNetTrainer_TCCT(nnUNetTrainer_ALUNET):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""

        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # !!!
        #self.num_epochs = 200
        #self.num_epochs = 500
        #self.num_epochs = 500
        self.num_epochs = 1000
        #self.num_iterations_per_epoch = 250
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 1
        self.select_only_labeled_train = True
        self.select_only_labeled_valid = False
        self.save_every = 50

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        
        #print('perform_actual_validation123')
        #sys.exit()

        # predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        #                             perform_everything_on_gpu=True, device=self.device, verbose=False,
        #                             verbose_preprocessing=False, allow_tqdm=False)
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_gpu=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=True)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)
        
        default_num_processes = 1
        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for k in dataset_val.keys():
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                     allowed_num_queued=2)
                    while not proceed:
                        sleep(0.1)
                        proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                         allowed_num_queued=2)
    
                    self.print_to_log_file(f"predicting {k}")
                    #print('k123', k)
                    data, seg, properties = dataset_val.load_case(k)
                    
                    #data = np.load('/mnt/HHD/data/UNetAL/LITS/AL/nnunet/nnUNet_preprocessed/Dataset106_ALUNet/nnUNetPlans_3d_fullres/volume-17.npz')['data']
                    #im = np.load('/mnt/HHD/data/UNetAL/LITS/AL/nnunet/nnUNet_preprocessed/Dataset106_ALUNet/nnUNetPlans_3d_fullres/volume-101.npy')
                    #seg = np.load('/mnt/HHD/data/UNetAL/LITS/AL/nnunet/nnUNet_preprocessed/Dataset106_ALUNet/nnUNetPlans_3d_fullres/volume-101_seg.npy')
                    #print('data123', data.shape)
                    #print('data123', data)
                    # print('seg123', seg.shape)
                    #print('seg123', seg.)
                    
                    
    
                    if self.is_cascaded:
                        data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                            output_dtype=data.dtype)))
                    with warnings.catch_warnings():
                        # ignore 'The given NumPy array is not writable' warning
                        warnings.simplefilter("ignore")
                        data = torch.from_numpy(data)
    
                    output_filename_truncated = join(validation_output_folder, k)
    
                    try:
                        prediction = predictor.predict_sliding_window_return_logits(data)
                        #print('prediction123', prediction.shape)
                        #print('prediction123', prediction)
                    except RuntimeError:
                        predictor.perform_everything_on_gpu = False
                        prediction = predictor.predict_sliding_window_return_logits(data)
                        predictor.perform_everything_on_gpu = True
                        
                    #im = np.load('/mnt/HHD/data/UNetAL/LITS/AL/nnunet/nnUNet_preprocessed/Dataset106_ALUNet/nnUNetPlans_3d_fullres/volume-103.npy')
                    #sys.exit()
                    prediction = prediction.cpu()
                    print('prediction12345', prediction.shape)
    
                    # this needs to go into background processes
                    results.append(
                        segmentation_export_pool.starmap_async(
                            export_prediction_from_logits, (
                                (prediction, properties, self.configuration_manager, self.plans_manager,
                                  self.dataset_json, output_filename_truncated, save_probabilities),
                            )
                        )
                    )
                    
                    # !!!
                    #export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,self.dataset_json, output_filename_truncated, save_probabilities)
                    
                    # #!!!
                    # results.append(
                    #     segmentation_export_pool.starmap(
                    #         export_prediction_from_logits, (
                    #             (prediction, properties, self.configuration_manager, self.plans_manager,
                    #              self.dataset_json, output_filename_truncated, save_probabilities),
                    #         )
                    #     )
                    # )
                    
                    print('export_prediction_from_logits1234')
                    
                    # for debug purposes
                    # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                    #              output_filename_truncated, save_probabilities)
    
                    # if needed, export the softmax prediction for the next stage
                    if next_stages is not None:
                        for n in next_stages:
                            next_stage_config_manager = self.plans_manager.get_configuration(n)
                            expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                                next_stage_config_manager.data_identifier)
    
                            try:
                                # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                                tmp = nnUNetDataset(expected_preprocessed_folder, [k],
                                                    num_images_properties_loading_threshold=0)
                                d, s, p = tmp.load_case(k)
                            except FileNotFoundError:
                                self.print_to_log_file(
                                    f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                    f"Run the preprocessing for this configuration first!")
                                continue
    
                            target_shape = d.shape[1:]
                            output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                            output_file = join(output_folder, k + '.npz')
    
                            # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                            #                   self.dataset_json)
                            results.append(segmentation_export_pool.starmap_async(
                                resample_and_save, (
                                    (prediction, target_shape, output_file, self.plans_manager,
                                     self.configuration_manager,
                                     properties,
                                     self.dataset_json),
                                )
                            ))

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]), also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()
        