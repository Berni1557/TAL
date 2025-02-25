import torch
from typing import Union, Tuple, List
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
from torch import autocast
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss, DC_and_CE_loss_CACS
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP

import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
from os.path import join
import time
from helper.helper import splitFilePath

from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from torch import distributed as dist
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper


from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.inference.predict_from_raw_data import compute_steps_for_sliding_window
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision


from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
#from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes

class nnUNetTrainer_ALUNET(nnUNetTrainer):
#class nnUNetTrainer_ALUNET(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""

        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # !!!
        #self.num_epochs = 200
        #self.num_epochs = 500
        self.num_epochs = 1000
        #self.num_epochs = 10
        #self.num_iterations_per_epoch = 250
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 1
        self.select_only_labeled_train = True
        self.select_only_labeled_valid = False
        self.save_every = 50

    def get_dataloaders_alunet(self, data_load, unlabeled=True, batch_size=None):
        # self=net.model['unet']
        
       
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        #print('patch_size1234', patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
            
        #print('initial_patch_size123', initial_patch_size)

        # training pipeline
        # tr_transforms = self.get_training_transforms(
        #     patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
        #     order_resampling_data=3, order_resampling_seg=1,
        #     use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
        #     is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
        #     regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
        #     ignore_label=self.label_manager.ignore_label)

        # ul_transforms = self.get_training_transforms(
        #     patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
        #     order_resampling_data=3, order_resampling_seg=1,
        #     use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
        #     is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
        #     regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
        #     ignore_label=self.label_manager.ignore_label)
        
        # validation pipeline
        ul_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)
        
        # # validation pipeline
        # val_transforms = self.get_validation_transforms(deep_supervision_scales,
        #                                                 is_cascaded=self.is_cascaded,
        #                                                 foreground_labels=self.label_manager.foreground_labels,
        #                                                 regions=self.label_manager.foreground_regions if
        #                                                 self.label_manager.has_regions else None,
        #                                                 ignore_label=self.label_manager.ignore_label)

        #dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)
        #dl_tr, dl_val = self.get_plain_dataloaders_BF(initial_patch_size, dim, unlabeled)
        dl_ul = self.get_plain_dataloaders_alunet(initial_patch_size, dim, data_load, unlabeled, batch_size)
        #dl_tr.infinite=True
        #print('infinite1234', dl_tr.infinite)
        #sys.exit()

        allowed_num_processes = get_allowed_n_proc_DA()
        #print('allowed_num_processes123', allowed_num_processes)
        #sys.exit()
        
        if allowed_num_processes == 0 or not dl_ul.infinite:
            #mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            #mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
            mt_gen_ul = SingleThreadedAugmenter(dl_ul, ul_transforms)
            #mt_gen_train = SingleThreadedAugmenter(dl_ul, ul_transforms)
        else:
            # mt_gen_train = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_ul, transform=ul_transforms,
            #                                  num_processes=allowed_num_processes, num_cached=6, seeds=None,
            #                                  pin_memory=self.device.type == 'cuda', wait_time=0.02)

            mt_gen_ul = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_ul,
                                            transform=ul_transforms, num_processes=max(1, allowed_num_processes // 2),
                                            num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
                                            wait_time=0.02)
            
            # mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
            #                                transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
            #                                num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
            #                                wait_time=0.02)
        #return mt_gen_train, mt_gen_val
        #return mt_gen_train
        return mt_gen_ul

    def get_plain_dataloaders_alunet(self, initial_patch_size: Tuple[int, ...], dim: int, data_load: list, unlabeled: bool, batch_size: int):
        #dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        #dataset_tr = self.get_tr_and_val_datasets_BF()
        dataset_ul = self.get_ul_datasets_BF()

        if dim == 2:
            # dl_ul = nnUNetDataLoader2DBF(dataset_ul, self.batch_size,
            #                            initial_patch_size,
            #                            self.configuration_manager.patch_size,
            #                            self.label_manager,
            #                            oversample_foreground_percent=self.oversample_foreground_percent,
            #                            infinite=False,
            #                            sampling_probabilities=None, pad_sides=None,
            #                            data_load=data_load,
            #                            unlabeled=unlabeled)
            
            if batch_size is None:
                batch_size=self.batch_size
            
            dl_ul = nnUNetDataLoader2DALUNET(dataset_ul, batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None,
                                        data_load=data_load,
                                        unlabeled=unlabeled)
            
            # dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
            #                             self.configuration_manager.patch_size,
            #                             self.configuration_manager.patch_size,
            #                             self.label_manager,
            #                             oversample_foreground_percent=self.oversample_foreground_percent,
            #                             sampling_probabilities=None, pad_sides=None)
        else:
            # dl_ul = nnUNetDataLoader3D(dataset_ul, self.batch_size,
            #                            initial_patch_size,
            #                            self.configuration_manager.patch_size,
            #                            self.label_manager,
            #                            oversample_foreground_percent=self.oversample_foreground_percent,
            #                            infinite=False,
            #                            sampling_probabilities=None, pad_sides=None)
            
            if batch_size is None:
                batch_size=self.batch_size
                
            dl_ul = nnUNetDataLoader3DALUNET(dataset_ul, batch_size,
                                       self.configuration_manager.patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       infinite=False,
                                       sampling_probabilities=None, pad_sides=None,
                                       data_load=data_load,
                                       unlabeled=unlabeled)
            
            # dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size,
            #                             self.configuration_manager.patch_size,
            #                             self.configuration_manager.patch_size,
            #                             self.label_manager,
            #                             oversample_foreground_percent=self.oversample_foreground_percent,
            #                             sampling_probabilities=None, pad_sides=None)
        #return dl_tr, dl_val
        return dl_ul

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                #print('batch_id123', batch_id)
                train_outputs.append(self.train_step(next(self.dataloader_train)))
                #train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    #print('batch_id1234', batch_id, self.num_val_iterations_per_epoch)
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
        
    def on_train_start(self):
        
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(True)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
            self.print_to_log_file('unpacking done...')

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")
        
    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)
        
        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)
        

        allowed_num_processes = get_allowed_n_proc_DA()
        #print('allowed_num_processes123', allowed_num_processes)
        #sys.exit()
        
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr, transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
                                           transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                           num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
                                           wait_time=0.02)
        return mt_gen_train, mt_gen_val
    
    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2DSPARSE(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader2DSPARSE(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr = nnUNetDataLoader3DSPARSE(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None,
                                       select_only_labeled=self.select_only_labeled_train)
            dl_val = nnUNetDataLoader3DSPARSE(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None,
                                        select_only_labeled=self.select_only_labeled_valid)
        return dl_tr, dl_val

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
        
class nnUNetDataLoader2DALUNET(nnUNetDataLoaderBase):
    
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 infinite: bool = False,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False,
                 data_load: list = [],
                 unlabeled: bool = False):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager, oversample_foreground_percent, 
                         infinite, sampling_probabilities, pad_sides, probabilistic_oversampling)
        self.data_load = data_load
        #self.data_load_ID = [s.F['ID'] for s in data_load]
        self.unlabeled=unlabeled
        self.IDD=0
        self.current_key=None
        self.data_org=None
        self.seg_org=None
        self.properties=None
        self.shape_org=None
        self.has_ignore=True
        self.idx = np.argsort([s.F['imagename'] for s in self.data_load])
        self.data_sort = [self.data_load[i] for i in self.idx]
        print('nnUNetDataLoader2DALUNETXXX')

    def generate_train_batch(self):
        
        data_all=[]
        seg_all=[]
        idx_all=[]
        imagename_all=[]
        
        # Extract batch samples
        end = min(self.IDD+self.batch_size, len(self.data_load))
        for b in range(self.IDD, end):
            s=self.data_sort[b]
            if b==0 or s.F['imagename']!=self.data_sort[b-1].F['imagename']:
                #key = s.F['imagename'].split('.')[0]
                #key = splitFilePath(s.F['imagename'])[1]
                key = s.F['imagename']
                #print('key123', key)
                self.data_org, self.seg_org, self.properties = self._data.load_case(key)
                #self.data_org, _, self.properties = self._data.load_case(key)
                #DNameFULL = opts.DName
                #fp_full = join(nnUNet_preprocessed, opts.DName, 'nnUNetPlans_'+opts.configuration)
                #self.seg_org = np.load(join(fp_full, key+'_seg.npy', 'r'))
                
            selected_slice = s.F['lbs_res'][0]
            
            #print('F',s.F)
            
            data_sel = self.data_org[:, selected_slice]
            seg_sel = self.seg_org[:, selected_slice]
            
            shape = data_sel.shape[1:]
            dim = len(shape)
            
            #class_locations={self.annotated_classes_key: s.F['classLocations']}
            #class_locations
            #bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg=False, class_locations=class_locations, overwrite_class=False)
            class_locations = {self.annotated_classes_key: s.F['classLocations'][:,[0,2,3]]}
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg=False, class_locations=class_locations, overwrite_class=False)
            
            
            #class_locations = s.F['classLocations']
            #bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg=False, class_locations=None, overwrite_class=False)
            
            # print('shape123', shape)
            #print('bbox_lbs123', bbox_lbs)
            #print('bbox_ubs123', bbox_ubs)
              
            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
            
            #print('valid_bbox_lbs123', valid_bbox_lbs)
            #print('valid_bbox_ubs123', valid_bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data_sel.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])

            data = data_sel[this_slice]
            this_slice = tuple([slice(0, seg_sel.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg_sel[this_slice]
            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            
            #print('padding123', padding)
            #print('s123', s.F)

            datap = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            datap = np.expand_dims(datap, 0)
            data_all.append(datap)

            segp = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
            segp = np.expand_dims(segp, 0)
            seg_all.append(segp)
            idx_all.append(self.idx[b])
            imagename_all.append(s.F['imagename'])
            
        data_all = np.vstack(data_all)
        seg_all = np.vstack(seg_all)
        idx_all = np.vstack(idx_all)
        imagename_all = np.vstack(imagename_all)
        
        self.IDD = self.IDD + self.batch_size
        #print("Process timeXXX: ", (time.time() - start))
        
        return {'data': data_all, 'seg': seg_all, 'idx': idx_all, 'imagename': imagename_all}

    def generate_mask(self, s, mask):
        
        selected_slice = s.F['lbs_res'][0]
        mask_sel = mask[selected_slice,:,:,:]
        
        shape = mask_sel.shape[1:]
        dim = len(shape)
        
        #class_locations = {self.annotated_classes_key: s.F['classLocations']}
        class_locations = {self.annotated_classes_key: s.F['classLocations'][:,[0,2,3]]}
        bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg=False, class_locations=class_locations, overwrite_class=False)
        
        #bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg=False, class_locations=None, overwrite_class=False)
        #bbox_lbs, bbox_ubs = dataloader_train.data_loader.get_bbox(shape, force_fg=False, class_locations=None, overwrite_class=False)
        
        # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
        # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
        # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
        # later
        valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
        valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
        
        #print('s', s.F)
        #print('valid_bbox_lbs', valid_bbox_lbs)
        #print('valid_bbox_ubs', valid_bbox_ubs)
        #print('annotated_classes_key', self.annotated_classes_key)
        
        # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
        # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
        # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
        # remove label -1 in the data augmentation but this way it is less error prone)
        #this_slice = tuple([slice(0, seg_sel.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])

        this_slice = tuple([slice(0, mask_sel.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
        mask_out = mask_sel[this_slice]
        padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
        mask_out = np.pad(mask_out, ((0, 0), *padding), 'constant', constant_values=-1)
        mask_out = np.expand_dims(mask_out, 0)
        mask_out[mask_out==-1]=0
        mask_out = torch.from_numpy(mask_out)
        return mask_out
        
class nnUNetDataLoader2DSPARSE(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        
        #selected_keys = self.get_indices()
        if self.select_only_labeled:
            selected_class = self.annotated_classes_key
            selected_keys=[]
            while(len(selected_keys)<self.batch_size):
                index = np.random.choice(self.indices, 1, replace=True, p=self.sampling_probabilities)[0]
                #print('index123', index)
                ret = self._data[index]
                #print('ret123', ret['properties'])
                if len(ret['properties']['class_locations'][selected_class]) > 0:
                    selected_keys.append(index)
        else:
            selected_keys = self.get_indices()
        
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        
        #start = time.time()

        for j, current_key in enumerate(selected_keys):
       
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)
            data, seg, properties = self._data.load_case(current_key)
            case_properties.append(properties)
            
            #print('time0', time.time()-start)
            #a=np.unique(seg)
            #print('time000', time.time()-start)
            
            #print('seg123', np.unique(seg))
            #if list(np.unique(seg))!=[2]:
            if 0 in seg:
                # print('seg123', np.unique(seg))
                #print('force_fg123', force_fg)
                # print('data123', data.shape)
                # print('self.has_ignore', self.has_ignore)
                # print('self.annotated_classes_key', self.annotated_classes_key)
    
    
                # select a class/region first, then a slice where this class is present, then crop to that area
                if not force_fg:
                    if self.has_ignore:
                        #print('time01', time.time()-start)
                        selected_class_or_region = self.annotated_classes_key
                        #print('time02', time.time()-start)
                    else:
                        selected_class_or_region = None
                else:
                    # filter out all classes that are not present here
                    eligible_classes_or_regions = [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) > 0]
    
                    # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                    # strange formulation needed to circumvent
                    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                    tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                    if any(tmp):
                        if len(eligible_classes_or_regions) > 1:
                            eligible_classes_or_regions.pop(np.where(tmp)[0][0])
    
                    selected_class_or_region = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        len(eligible_classes_or_regions) > 0 else None
                if selected_class_or_region is not None:
                    #print('time1', time.time()-start)
                    #print('current_key123', current_key)
                    # print('selected_class_or_region123', selected_class_or_region)
                    # print('selected_class_or_region123', selected_class_or_region)
                    #print('propertiesclass_locations123', np.unique(properties['class_locations'][selected_class_or_region][:, 1]))
                    selected_slice = np.random.choice(properties['class_locations'][selected_class_or_region][:, 1])
                else:
                    selected_slice = np.random.choice(len(data[0]))
    
                data = data[:, selected_slice]
                seg = seg[:, selected_slice]
    
                # the line of death lol
                # this needs to be a separate variable because we could otherwise permanently overwrite
                # properties['class_locations']
                # selected_class_or_region is:
                # - None if we do not have an ignore label and force_fg is False OR if force_fg is True but there is no foreground in the image
                # - A tuple of all (non-ignore) labels if there is an ignore label and force_fg is False
                # - a class or region if force_fg is True
                class_locations = {
                    selected_class_or_region: properties['class_locations'][selected_class_or_region][properties['class_locations'][selected_class_or_region][:, 1] == selected_slice][:, (0, 2, 3)]
                } if (selected_class_or_region is not None) else None
    
                # print(properties)
                shape = data.shape[1:]
                dim = len(shape)
                bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else None,
                                                   class_locations, overwrite_class=selected_class_or_region)
    
                # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
                # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
                # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
                # later
                valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
                valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
    
                # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
                # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
                # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
                # remove label -1 in the data augmentation but this way it is less error prone)
                this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
                data = data[this_slice]
    
                this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
                seg = seg[this_slice]
    
                padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
                data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
                seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
                #print('time2', time.time()-start)
        #print('time0', time.time()-start)
        #print('data_all123', data_all.shape)
        
        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


class nnUNetDataLoader3DALUNET(nnUNetDataLoaderBase):


    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 infinite: bool = False,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False,
                 data_load: list = [],
                 unlabeled: bool = False):
        self.data_load=data_load
        self.unlabeled=unlabeled
        #print('data123', data)
        #print('patch_size123', patch_size)
        #print('final_patch_size123', final_patch_size)
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager, oversample_foreground_percent, 
                         infinite, sampling_probabilities, pad_sides, probabilistic_oversampling)
        self.data_load = data_load
        #self.data_load_ID = [s.F['ID'] for s in data_load]
        self.unlabeled=unlabeled
        self.IDD=0
        self.current_key=None
        self.data_org=None
        self.seg_org=None
        self.properties=None
        self.shape_org=None
        self.has_ignore=True
        self.idx = np.argsort([s.F['imagename'] for s in self.data_load])
        self.data_sort = [self.data_load[i] for i in self.idx]
        print('nnUNetDataLoader3DALUNETXXX')


    def generate_train_batch(self):
        #print('generate_train_batch123')

        
        data_all=[]
        seg_all=[]
        idx_all=[]
        imagename_all=[]
        
        # Extract batch samples
        end = min(self.IDD+self.batch_size, len(self.data_load))
        for b in range(self.IDD, end):
            s=self.data_sort[b]
            if b==0 or s.F['imagename']!=self.data_sort[b-1].F['imagename']:
                #key = s.F['imagename'].split('.')[0]
                #key = splitFilePath(s.F['imagename'])[1]
                key = s.F['imagename']
                #print('key123', key)
                self.data_org, self.seg_org, self.properties = self._data.load_case(key)
                #self.data_org, _, self.properties = self._data.load_case(key)
                #DNameFULL = opts.DName
                #fp_full = join(nnUNet_preprocessed, opts.DName, 'nnUNetPlans_'+opts.configuration)
                #self.seg_org = np.load(join(fp_full, key+'_seg.npy', 'r'))
                
            #selected_slice = s.F['lbs_res'][0]
            
            #print('F',s.F)
            
            #data_sel = self.data_org[:, selected_slice]
            #seg_sel = self.seg_org[:, selected_slice]
            
            data_sel = self.data_org
            seg_sel = self.seg_org
            
            shape = data_sel.shape[1:]
            dim = len(shape)
            
            #class_locations={self.annotated_classes_key: s.F['classLocations']}
            #class_locations
            #bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg=False, class_locations=class_locations, overwrite_class=False)
            #class_locations = {self.annotated_classes_key: s.F['classLocations'][:,[0,2,3]]}
            class_locations = {self.annotated_classes_key: s.F['classLocations'][:,[0,1,2,3]]}
            #print('classLocations123', s.F['classLocations'])
            #print('class_locations123', class_locations)
            
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg=False, class_locations=class_locations, overwrite_class=False)
            
            
            #class_locations = s.F['classLocations']
            #bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg=False, class_locations=None, overwrite_class=False)
            
            # print('shape123', shape)
            #print('bbox_lbs123', bbox_lbs)
            #print('bbox_ubs123', bbox_ubs)
              
            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
            
            #print('valid_bbox_lbs123', valid_bbox_lbs)
            #print('valid_bbox_ubs123', valid_bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data_sel.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])

            data = data_sel[this_slice]
            this_slice = tuple([slice(0, seg_sel.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg_sel[this_slice]
            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            
            #print('padding123', padding)
            #print('s123', s.F)

            datap = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            datap = np.expand_dims(datap, 0)
            data_all.append(datap)

            segp = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
            segp = np.expand_dims(segp, 0)
            seg_all.append(segp)
            idx_all.append(self.idx[b])
            imagename_all.append(s.F['imagename'])
            
        data_all = np.vstack(data_all)
        seg_all = np.vstack(seg_all)
        idx_all = np.vstack(idx_all)
        imagename_all = np.vstack(imagename_all)
        
        self.IDD = self.IDD + self.batch_size
        #print("Process timeXXX: ", (time.time() - start))
        
        return {'data': data_all, 'seg': seg_all, 'idx': idx_all, 'imagename': imagename_all}

class nnUNetDataLoader3DSPARSE(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        
        if self.select_only_labeled:
            selected_class = self.annotated_classes_key
            selected_keys=[]
            while(len(selected_keys)<self.batch_size):
                index = np.random.choice(self.indices, 1, replace=True, p=self.sampling_probabilities)[0]
                #print('index123', index)
                ret = self._data[index]
                #print('ret123', ret['properties'])
                if len(ret['properties']['class_locations'][selected_class]) > 0:
                    selected_keys.append(index)
        else:
            selected_keys = self.get_indices()

        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        
        #print('patch_size1236', self.patch_size)
        #print('final_patch_size1236', self.final_patch_size)
        #sys.exit()

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            #print('j', j)
            #print('i', i)
            ret = self._data[i]
            #print('ret', ret['properties'])
            
            selected_class = self.annotated_classes_key

        
            force_fg = self.get_do_oversample(j)
            #print('force_fg123', force_fg)

            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            
            #print('bbox_lbs123', bbox_lbs)
            #print('bbox_ubs123', bbox_ubs)
            
            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]
            
            #print('seg123', seg.shape)
            #print('this_slice123', this_slice)

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
            
            #print('seg_all1236', np.unique(seg_all))
            #print('data_all1236', data_all.shape)
            #sys.exit()

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}

