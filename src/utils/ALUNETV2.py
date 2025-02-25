#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:59:21 2024

@author: bernifoellmer
"""
import os, sys
from os.path import join
import shutil
import subprocess
import torch
import math
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from glob import glob
import pandas as pd
import random
import json
import pickle
from collections import defaultdict
from distutils.dir_util import copy_tree
from utils.ALManager import ALManager
from strategies.RANDOM import RandomStrategy
from utils.ALSample import ALSample
from utils.ALManager import SALDataset
from utils.ct import CTImage, CTRef
from nnunet.code.nnunetv2.run.run_training import run_training
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.inference.predict_from_raw_data import compute_steps_for_sliding_window, nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from utils.DLBaseModel import DLBaseModel
import matplotlib.pyplot as plt
from utils.helper import compute_one_hot_torch
from utils.ALAction import ALAction
import zipfile
from collections import OrderedDict
from utils.helper import splitFilePath
import skimage
from utils.metrices_new import confusion, f1
from strategies.MCD import MCD
from strategies.UCPROP import UCPROP
from strategies.USIMF import USIMF
from strategies.USIMFT import USIMFT
from strategies.BADGE import BADGE

soft1 = torch.nn.Softmax(dim=1)

strategies={'RANDOM':RandomStrategy,
            'USIMF': USIMF,
            'USIMFT': USIMFT,
            'UCPROP': UCPROP,
            'MCD': MCD,
            'USIMFTRG': USIMFT,
            'USIMFTLG': USIMFT,
            'USIMFTSIDE': USIMFT,
            'BADGE': BADGE}

class ALUNETV2():
    
    def __init__(self, opts):
        opts_dict = vars(opts)
        opts_dict['CLDataset'] = UNetDatasetV2
        opts_dict['CLSample'] = UNetSampleV2
        opts_dict['CLPatch'] = UNetPatchV2
        opts_dict['CLManager'] = UNetManagerV2
        opts_dict['fp_images'] = ''
        opts_dict['fp_references'] = ''  
        opts_dict['fp_references_org'] = '' 
        opts_dict['nnUNetTrainer'] = opts.nnUNetTrainer 
        opts_dict['plans_identifier'] = 'ALUNETPlanner'
        opts_dict['DName'] = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        opts_dict['DNameFull'] = 'Dataset' + opts.dataset_name_or_id[0] + '01_' + opts.dataset

    def init_dataset(self, opts):
        if opts.targeted:
            self.init_dataset_target(opts)
        else:
            self.init_dataset_plain(opts) 

    def init_dataset_target(self, opts):
        
        # self=alunet
        
        # Init dataset
        method = 'INIT'
        NewVersion = False
        NumSamples = opts.ALSamples[0]
        NumSamplesPre = NumSamples*3
        self.man = opts.CLManager(fp_dataset='')
        opts.alunet = self
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=method, NewVersion=NewVersion)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round', 'action_pre'], load_class=opts.CLPatch, hdf5=False)
        if opts.fp_manual is None:
            fp_manual = join(os.path.dirname(self.man.folderDict['modelpath']), 'data_manual')
        else:
            fp_manual = opts.fp_manual
        fip_actionlist = os.path.join(fp_manual, 'actionlist.json')
                
        # Set parameter
        # NumSamplesCheck = 50
        NumSamplesPre = NumSamples*3
        
        # Check if samples are unlabeled
        if opts.label_manual:
            #if not self.man.folderDict['round_status']['correction_proposal']:
            
            if not folderDict['round_status']['correction_proposal']:
                # Convert to nnUNet structure
                self.convert_raw(opts, delete_label=True)
            
                # Preprocessing raw data
                subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -overwrite_plans_name " + opts.plans_identifier +  " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
                                
                # Init nnUNet datasets
                self.man.init_patches(opts)
                self.man.save(save_dict=dict(), save_class=opts.CLPatch, hdf5=False)
                
                # Label validation set
                if opts.label_valid:
                    self.annotation_auto(opts, self.man, data=self.man.datasets['valid'].data, full=True)
                
                # Set correction flags
                self.man.update_status(folderDict, 'correction_proposal', True)
            
            # Selection proposal
            if not folderDict['round_status']['selection_proposal']:
                # Query new samples
                strat = opts.strategy
                opts.strategy='RANDOM'
                self.query(opts, self.man.folderDict, self.man, NumSamples=NumSamplesPre, NumSamplesMaxMCD=5000)
                opts.strategy = strat
                self.man.datasets['action_pre'].data = self.man.datasets['action_round'].data
                self.man.datasets['action_round'].data=[]
                self.man.save(include=['action_round', 'action_pre'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
                # Create action
                self.create_action(opts, self.man, self.man.datasets['action_pre'].data, classification=True, classification_multislice=True, create_mask=False, create_pseudo=False, pseudo_full=True)
                self.man.update_status(folderDict, 'selection_proposal', True)
            
    
            # Selection of targets (manual)
            if not folderDict['round_status']['selection_target_manual']:
                sys.exit('Please classifie images in positive and negative using XALabeler software!')
                self.man.update_status(folderDict, 'selection_target_manual', True)
                    
            # Subset selection based on selected target samples
            if not folderDict['round_status']['selection_subset']:
                if opts.dim==2:
                    # Read action list
                    fip_actionlist = os.path.join(fp_manual, 'actionlist.json')
                    actionlist = ALAction.load(fip_actionlist) 
                    action_pre = self.man.datasets['action_pre'].data
                    for a in actionlist:
                        for s in action_pre:
                            if a.id==s.ID:
                                s.action = a
                                a.filetype='.nii.gz'
                    action_pre = [s for s in action_pre if hasattr(s, "action")]
                    
                    # Check if samples are classified as positive and negative
                    classes = [(s.F['imagename'], s.action.info['class']) for s in action_pre if 'class' in s.action.info]
                    df_classes = pd.DataFrame()
                    for cl in classes:
                        imname = cl[0]
                        for sl in cl[1]:
                            df_classes=df_classes.append({'imagename': imname, 'slice': sl[0], 'class': sl[1]}, ignore_index=True)
                    
                    data_query = self.man.datasets['query'].data
                    data_action_select = []
                    for index, row in df_classes.iterrows():
                        for s in action_pre+data_query:
                            if (s.F['imagename']==row['imagename']) and (s.F['lbs_org'][0]==row['slice']):
                                s.action = ALAction()
                                s.action.info['class']=row['class']
                                if row['class']==1:
                                    data_action_select.append(s)
                    # Select random subset
                    idx = [x for x in range(len(data_action_select))]
                    random.shuffle(idx)
                    idx = idx[0:NumSamples]
                    data_action_select = [data_action_select[i] for i in idx]
                else:
                    data = self.man.datasets['action_pre'].data + self.man.datasets['query'].data
                    file1 = open(os.path.join(fp_manual, 'XALabelerLUT.ctbl'), 'r')
                    lines = file1.readlines()
                    for line in lines:
                        if line.split(' ')[1]=='positive':
                            label_positive=int(line.split(' ')[0])
                        if line.split(' ')[1]=='negative':
                            label_negative=int(line.split(' ')[0])
                            
                    imagenames = sorted(np.unique([s.F['imagename'] for s in data]))
                    data_action_select=[]
                    for i,imagename in enumerate(imagenames):
                        #print(i)
                        fip_label = join(fp_manual, 'refine', imagename + '.nii.gz')
                        if os.path.isfile(fip_label):
                            ref = CTRef(fip_label).ref()
                            label_pos = skimage.measure.label(ref==label_positive)
                            props_pos = skimage.measure.regionprops_table(label_pos, properties=['centroid', 'area', 'label'])
                            idx_pos = np.ones((props_pos['centroid-0'].shape[0]))*-1
                            dist_pos = np.ones((props_pos['centroid-0'].shape[0]))*1000000
                            label_neg = skimage.measure.label(ref==label_negative)
                            props_neg = skimage.measure.regionprops_table(label_neg, properties=['centroid', 'area', 'label'])                      
                            idx_neg = np.ones((props_neg['centroid-0'].shape[0]))*-1
                            dist_neg = np.ones((props_neg['centroid-0'].shape[0]))*1000000
                            for i,s in enumerate(data):
                                if s.F['imagename']==imagename:
                                    bL=s.F['lbs_org']
                                    bU=s.F['ubs_org']
                                    x = np.mean([bL[0],bU[0]])
                                    y = np.mean([bL[1],bU[1]])
                                    z = np.mean([bL[2],bU[2]])
                                    # Positive samples
                                    distp = ((x-props_pos['centroid-0'])*(x-props_pos['centroid-0']))+((y-props_pos['centroid-1'])*(y-props_pos['centroid-1']))+((z-props_pos['centroid-2'])*(z-props_pos['centroid-2']))
                                    idxp = np.where(distp<dist_pos)
                                    dist_pos[idxp] = distp[idxp]
                                    idx_pos[idxp] = i
                                    # Negative samples
                                    distn = ((x-props_neg['centroid-0'])*(x-props_neg['centroid-0']))+((y-props_neg['centroid-1'])*(y-props_neg['centroid-1']))+((z-props_neg['centroid-2'])*(z-props_neg['centroid-2']))
                                    idxn = np.where(distn<dist_neg)
                                    dist_neg[idxn] = distn[idxn]
                                    idx_neg[idxn] = i
                            
                            idx_pos = np.unique(idx_pos)
                            idx_neg = np.unique(idx_neg)
                            for i in list(idx_pos):
                                s = data[int(i)]
                                s.action = ALAction()
                                s.action.info['class']=1
                                data_action_select.append(s)
                            for i in list(idx_neg):
                                s = data[int(i)]
                                s.action = ALAction()
                                s.action.info['class']=2
                                data_action_select.append(s)

                #data_select = self.select_manual(opts, self.man.folderDict, self.man, data_action_select, NumSamples=NumSamples)
                data_select = random.sample(data_action_select, k=min(len(data_action_select), NumSamples))
                self.man.datasets['query'].delete(data_select)
                #self.man.datasets['action_pre'].data = self.man.datasets['action_round'].data
                self.man.datasets['action_round'].data = data_select
                self.man.save(include=['action_round', 'train','query'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
                # Create action
                self.create_action(opts, self.man, self.man.datasets['action_round'].data, classification=False, create_mask=False, create_pseudo=False)
                self.man.update_status(self.man.folderDict, 'selection_subset', True)


            # Annotated images (manual)
            if not self.man.folderDict['round_status']['selection_annotation_manual']:
                print('Doing selection_annotation_manual')
                if opts.segauto:
                    self.annotation_auto(opts, self.man, data=self.man.datasets['action_round'].data, full=False)
                    self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)
                else:
                    sys.exit('Please annotate images using XALabeler software!')
                    #_ = self.annotation_manual(opts, self.man, self.man.datasets['action_round'].data, bg_offset=False)
                    #self.annotation_auto(opts, self.man, data=self.man.datasets['action_round'].data, full=False)
                    #sys.exit('Please annotate images using XALabeler software!')
                    self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)

            if not self.man.folderDict['round_status']['training']:
                if not opts.segauto:
                    _ = self.annotation_manual(opts, self.man, self.man.datasets['action_round'].data, bg_offset=False)
                    # Preprocessing raw data
                    #subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
                subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -overwrite_plans_name " + opts.plans_identifier +  " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
                # Train model
                self.train(opts)
                # Copy model and labeles
                self.copy_model_labels(opts, self.man.folderDict)
                self.man.update_status(self.man.folderDict, 'training', True)
              

        
    def init_dataset_plain(self, opts):
        
        # self=alunet
        
        # Init dataset
        method = 'INIT'
        NewVersion = False
        NumSamples = opts.ALSamples[0]
        self.man = opts.CLManager(fp_dataset='')
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=method, NewVersion=NewVersion)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        if opts.fp_manual is None:
            fp_manual = join(os.path.dirname(self.man.folderDict['modelpath']), 'data_manual')
        else:
            fp_manual = opts.fp_manual
        fip_actionlist = os.path.join(fp_manual, 'actionlist.json')
            
        # Check if samples are unlabeled
        if opts.label_manual:
            if not self.man.folderDict['round_status']['selection_subset']:
                # Convert to nnUNet structure
                self.convert_raw(opts, delete_label=True)
            
                # Preprocessing raw data
                #subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
                subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -overwrite_plans_name " + opts.plans_identifier +  " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
                
                # Init nnUNet datasets
                self.man.init_patches(opts)
            
                # Init train set
                self.man.datasets['action_round'].data = self.man.getRandom(dataset='query', NumSamples=NumSamples, remove=True)
                self.man.save(save_dict=dict(), save_class=opts.CLPatch, hdf5=False)
                self.create_action(opts, self.man, self.man.datasets['action_round'].data, create_pseudo=False)
                
                # Annotate validation set
                _ = self.annotation_auto(opts, self.man, data=self.man.datasets['valid'].data, full=True)
                
                self.man.update_status(self.man.folderDict, 'correction_proposal', True)
                self.man.update_status(self.man.folderDict, 'correction_selection_manual', True)
                self.man.update_status(self.man.folderDict, 'correction_preparation', True)
                self.man.update_status(self.man.folderDict, 'correction_annotation_manual', True)
                self.man.update_status(self.man.folderDict, 'selection_proposal', True)
                self.man.update_status(self.man.folderDict, 'selection_target_manual', True)
                self.man.update_status(self.man.folderDict, 'selection_subset', True)
                
            if not self.man.folderDict['round_status']['selection_annotation_manual']:
                annotated = self.annotation_manual(opts, self.man, data_action=self.man.datasets['action_round'].data)
                if not annotated:
                    sys.exit('Please annotate images using XALabeler software!')
                self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)
                #self.annotation_auto(opts, self.man, data=self.man.datasets['train'].data)

            if not self.man.folderDict['round_status']['training']:
                  # Preprocessing raw data
                  #subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
                  subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -overwrite_plans_name " + opts.plans_identifier +  " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
                  # Train model
                  self.train(opts)
                  # Copy model and labeles
                  self.copy_model_labels(opts, self.man.folderDict)
                  self.man.update_status(self.man.folderDict, 'training', True)
               
        else:

            # Convert to nnUNet structure
            self.convert_raw(opts, delete_label=True)
        
            # Preprocessing raw data
            subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -overwrite_plans_name " + opts.plans_identifier +  " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            
            # Init nnUNet datasets
            self.man.init_patches(opts)
        
            # Init train set
            self.man.datasets['train'].data = self.man.getRandom(dataset='query', NumSamples=NumSamples, remove=True)
            self.man.save(save_dict=dict(), save_class=opts.CLPatch, hdf5=False)
            
            # Update labels
            self.annotation_auto(opts, self.man, data=self.man.datasets['train'].data)
            self.annotation_auto(opts, self.man, data=self.man.datasets['valid'].data, full=True)

            # Preprocessing raw data
            #subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -overwrite_plans_name " + opts.plans_identifier +  " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            
            # Train model
            self.train(opts)
            
            # Copy model and labeles
            self.copy_model_labels(opts, folderDict)
            
            # Update state
            self.man.update_status(self.man.folderDict, 'correction_proposal', True)
            self.man.update_status(self.man.folderDict, 'correction_selection_manual', True)
            self.man.update_status(self.man.folderDict, 'correction_preparation', True)
            self.man.update_status(self.man.folderDict, 'correction_annotation_manual', True)
            self.man.update_status(self.man.folderDict, 'selection_proposal', True)
            self.man.update_status(self.man.folderDict, 'selection_target_manual', True)
            self.man.update_status(self.man.folderDict, 'selection_subset', True)
            self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)
            self.man.update_status(self.man.folderDict, 'training', True)
            

    def copy_model_labels(self, opts, folderDict):
        # Copy model back
        fp_results = os.path.join(opts.fp_nnunet, 'nnUNet_results', opts.DName, opts.nnUNetResults)
        fp_results_new = os.path.join(folderDict['modelpath'], opts.nnUNetResults)
        shutil.copytree(fp_results, fp_results_new, dirs_exist_ok=True)
        
        # Copy labels into model folder
        fp_labelsTr = os.path.join(join(nnUNet_raw, opts.DName), 'labelsTr')
        fp_labelsTr_new = os.path.join(folderDict['modelpath'], 'labelsTr')
        shutil.copytree(fp_labelsTr, fp_labelsTr_new, dirs_exist_ok=True)
        

    def convert_raw(self, opts, delete_label=True):
        
        foldername = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        
        # setting up nnU-Net folders
        out_base = os.path.join(nnUNet_raw, foldername)
        imagestrTr = os.path.join(out_base, "imagesTr")
        labelstrTr = os.path.join(out_base, "labelsTr")
        imagestrTs = os.path.join(out_base, "imagesTe")
        labelstrTs = os.path.join(out_base, "labelsTe")
        maybe_mkdir_p(imagestrTr)
        maybe_mkdir_p(labelstrTr)
        maybe_mkdir_p(imagestrTs)
        maybe_mkdir_p(labelstrTs)
        
        if opts.dataset=='BrainTumour':
            label_ignore = 4
            # Copy imagesTr
            fip_images = glob(join(opts.fp_raw, opts.dataset, 'imagesTr') + '/*')
            num_training_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTr'):
                im = sitk.ReadImage(fip_image)
                arr = sitk.GetArrayFromImage(im)
                if len(arr.shape)==3:
                    sitk.WriteImage(im, join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_0000.nrrd'), True)
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        #imc.SetDirection(im.GetDirection())
                        sitk.WriteImage(imc, join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nrrd'), True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy imagesTs
            fip_images = glob(join(opts.fp_raw, opts.dataset, 'imagesTs') + '/*')
            num_test_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTs'):
                im = sitk.ReadImage(fip_image)
                arr = sitk.GetArrayFromImage(im)
                if len(arr.shape)==3:
                    sitk.WriteImage(im, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_0000.nrrd'), True)
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        #imc.SetDirection(im.GetDirection())
                        sitk.WriteImage(imc, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nrrd'), True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy labelTr
            fip_labels = glob(join(opts.fp_raw, opts.dataset, 'labelsTr') + '/*')
            for fip_label in tqdm(fip_labels, desc='Copy labelsTr'):
                fip_image = join(imagestrTr, os.path.basename(fip_label).split('.')[0]+'_' + '0000' + '.nrrd')
                iml = sitk.ReadImage(fip_label)
                im = sitk.ReadImage(fip_image)
                if delete_label:
                    arr = sitk.GetArrayFromImage(iml)
                    arr[:] = label_ignore
                    iml = sitk.GetImageFromArray(arr)
                    iml.SetOrigin(im.GetOrigin())
                    iml.SetSpacing(im.GetSpacing())
                    iml.SetDirection(im.GetDirection())
                else:
                    iml = iml
                sitk.WriteImage(iml, join(labelstrTr, os.path.basename(fip_label).split('.')[0]+'.nrrd'), True)
    
            # Copy labelTe
            fip_labels = glob(join(opts.fp_raw, opts.dataset, 'labelsTs') + '/*')
            for fip_label in tqdm(fip_labels, desc='Copy labelsTs'):
                im = sitk.ReadImage(fip_label)
                sitk.WriteImage(im, join(labelstrTs, os.path.basename(fip_label).split('.')[0]+'.nrrd'), True)
    
            # Generate json file of dataset
            generate_dataset_json(out_base, {0: "FLAIR", 1: "T1w", 2: "t1gd", 3: "T2w"},
                                  labels={
                                      "background": 0,
                                      "edema": 1,
                                      "non-enhancing tumor": 2,
                                      "enhancing tumour": 3,
                                      "ignore": label_ignore
                                  },
                                  regions_class_order=(1, 2, 3),
                                  num_training_cases=num_training_cases,
                                  num_test_cases=num_test_cases,
                                  file_ending='.nrrd',
                                  dataset_name=opts.dataset, reference='none',
                                  release='prerelease',
                                  overwrite_image_reader_writer='SimpleITKIO',
                                  description=opts.dataset)
        elif opts.dataset=='Liver':
            label_ignore = 3
            # Copy imagesTr
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTr') + '/*'))
            num_training_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTr'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    im.save(join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        fip_image_out = join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz')
                        sitk.WriteImage(imc, fip_image_out, True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy imagesTs
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTs') + '/*'))
            num_test_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTs'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    #sitk.WriteImage(im, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_0000.mhd'), True)
                    im.save(join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        #imc.SetDirection(im.GetDirection())
                        sitk.WriteImage(imc, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz'), True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy labelTr
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTr') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTr'):
                fip_image = join(imagestrTr, os.path.basename(fip_label).split('.')[0]+'_' + '0000' + '.nii.gz')
                iml = sitk.ReadImage(fip_label)
                im = sitk.ReadImage(fip_image)
                if delete_label:
                    arr = sitk.GetArrayFromImage(iml)
                    arr[:] = label_ignore
                    iml = sitk.GetImageFromArray(arr)
                    iml.SetOrigin(im.GetOrigin())
                    iml.SetSpacing(im.GetSpacing())
                    iml.SetDirection(im.GetDirection())
                else:
                    iml = iml
                sitk.WriteImage(iml, join(labelstrTr, os.path.basename(fip_label).split('.')[0]+'.nii.gz'), useCompression=True)
    
            # Copy labelTe
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTs') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTs'):
                im = sitk.ReadImage(fip_label)
                sitk.WriteImage(im, join(labelstrTs, os.path.basename(fip_label).split('.')[0]+'.nii.gz'), True)
    
            # Generate json file of dataset
            generate_dataset_json(out_base, {0: "CT"},
                                  labels={
                                      "background": 0,
                                      "liver": 1,
                                      "cancer": 2,
                                      "ignore": label_ignore
                                  },
                                  regions_class_order=(1, 2, 3),
                                  num_training_cases=num_training_cases,
                                  num_test_cases=num_test_cases,
                                  file_ending='.nii.gz',
                                  dataset_name=opts.dataset, reference='none',
                                  release='prerelease',
                                  overwrite_image_reader_writer='SimpleITKIO',
                                  description=opts.dataset)
        elif opts.dataset=='Spleen':
            label_ignore = 2
            # Copy imagesTr
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTr') + '/*'))
            num_training_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTr'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    im.save(join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        fip_image_out = join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz')
                        sitk.WriteImage(imc, fip_image_out, True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy imagesTs
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTs') + '/*'))
            num_test_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTs'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    #sitk.WriteImage(im, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_0000.mhd'), True)
                    im.save(join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        #imc.SetDirection(im.GetDirection())
                        sitk.WriteImage(imc, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz'), True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy labelTr
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTr') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTr'):
                fip_image = join(imagestrTr, os.path.basename(fip_label).split('.')[0]+'_' + '0000' + '.nii.gz')
                iml = sitk.ReadImage(fip_label)
                im = sitk.ReadImage(fip_image)
                if delete_label:
                    arr = sitk.GetArrayFromImage(iml)
                    arr[:] = label_ignore
                    iml = sitk.GetImageFromArray(arr)
                    iml.SetOrigin(im.GetOrigin())
                    iml.SetSpacing(im.GetSpacing())
                    iml.SetDirection(im.GetDirection())
                else:
                    iml = iml
                sitk.WriteImage(iml, join(labelstrTr, os.path.basename(fip_label).split('.')[0]+'.nii.gz'), useCompression=True)
        
            # Copy labelTe
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTs') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTs'):
                im = sitk.ReadImage(fip_label)
                sitk.WriteImage(im, join(labelstrTs, os.path.basename(fip_label).split('.')[0]+'.nii.gz'), True)
        
            # Generate json file of dataset
            generate_dataset_json(out_base, {0: "CT"},
                                  labels={
                                      "background": 0,
                                      "spleen": 1,
                                      "ignore": label_ignore
                                  },
                                  regions_class_order=(1, 2, 3),
                                  num_training_cases=num_training_cases,
                                  num_test_cases=num_test_cases,
                                  file_ending='.nii.gz',
                                  dataset_name=opts.dataset, reference='none',
                                  release='prerelease',
                                  overwrite_image_reader_writer='SimpleITKIO',
                                  description=opts.dataset)
        elif opts.dataset=='ULS':
            label_ignore = 2
            
            # Filter cases with images and labels
            images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTr') + '/*'))
            images = [os.path.basename(x).split('.')[0] for x in images]
            # Filter subset
            images = [im for im in images if ('kidney' in im or 'liver' in im or 'lung' in im)]
            
            labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTr') + '/*'))
            labels = [os.path.basename(x).split('.')[0] for x in labels]
            
            images = [x for x in images if x in labels]
            labels = [x for x in labels if x in images]
            
            # Copy imagesTr
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTr') + '/*'))
            fip_images = [x for x in fip_images if os.path.basename(x).split('.')[0] in images]
            num_training_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTr'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    im.save(join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        fip_image_out = join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz')
                        sitk.WriteImage(imc, fip_image_out, True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            
            
            # Copy labelTr
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTr') + '/*'))
            fip_labels = [x for x in fip_labels if os.path.basename(x).split('.')[0] in labels]
            for fip_label in tqdm(fip_labels, desc='Copy labelsTr'):
                if fip_label[-3:]=='zip':
                    archive = zipfile.ZipFile(fip_label, 'r')
                    archive.extract(archive.namelist()[0], path=join(opts.fp_raw, opts.dataset, 'labelsTr'))
                    fip_label_unzip = join(opts.fp_raw, opts.dataset, 'labelsTr', archive.namelist()[0])
                    os.remove(fip_label)
                else:
                    fip_label_unzip = fip_label
                #fip_label_tmp = temp_dir + '/' + archive.namelist()[0]
               
                fip_image = join(imagestrTr, os.path.basename(fip_label).split('.')[0]+'_' + '0000' + '.nii.gz')
                iml = sitk.ReadImage(fip_label_unzip)
                im = sitk.ReadImage(fip_image)
                if delete_label:
                    arr = sitk.GetArrayFromImage(iml)
                    arr[:] = label_ignore
                    iml = sitk.GetImageFromArray(arr)
                    iml.SetOrigin(im.GetOrigin())
                    iml.SetSpacing(im.GetSpacing())
                    iml.SetDirection(im.GetDirection())
                else:
                    iml = iml
                sitk.WriteImage(iml, join(labelstrTr, os.path.basename(fip_label).split('.')[0]+'.nii.gz'), useCompression=True)
                
            
            # Generate json file of dataset
            generate_dataset_json(out_base, {0: "CT"},
                                  labels={
                                      "background": 0,
                                      "lesion": 1,
                                      "ignore": label_ignore
                                  },
                                  regions_class_order=(1, 2, 3),
                                  num_training_cases=num_training_cases,
                                  num_test_cases=0,
                                  file_ending='.nii.gz',
                                  dataset_name=opts.dataset, reference='none',
                                  release='prerelease',
                                  overwrite_image_reader_writer='SimpleITKIO',
                                  description=opts.dataset)

        elif opts.dataset=='SpleenX':
            label_ignore = 2
            # Copy imagesTr
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTr') + '/*'))
            num_training_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTr'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    im.save(join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        fip_image_out = join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz')
                        sitk.WriteImage(imc, fip_image_out, True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy imagesTs
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTs') + '/*'))
            num_test_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTs'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    #sitk.WriteImage(im, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_0000.mhd'), True)
                    im.save(join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        #imc.SetDirection(im.GetDirection())
                        sitk.WriteImage(imc, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz'), True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy labelTr
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTr') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTr'):
                fip_image = join(imagestrTr, os.path.basename(fip_label).split('.')[0]+'_' + '0000' + '.nii.gz')
                iml = sitk.ReadImage(fip_label)
                im = sitk.ReadImage(fip_image)
                if delete_label:
                    arr = sitk.GetArrayFromImage(iml)
                    arr[:] = label_ignore
                    iml = sitk.GetImageFromArray(arr)
                    iml.SetOrigin(im.GetOrigin())
                    iml.SetSpacing(im.GetSpacing())
                    iml.SetDirection(im.GetDirection())
                else:
                    iml = iml
                sitk.WriteImage(iml, join(labelstrTr, os.path.basename(fip_label).split('.')[0]+'.nii.gz'), useCompression=True)
        
            # Copy labelTe
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTs') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTs'):
                im = sitk.ReadImage(fip_label)
                sitk.WriteImage(im, join(labelstrTs, os.path.basename(fip_label).split('.')[0]+'.nii.gz'), True)
        
            # Generate json file of dataset
            generate_dataset_json(out_base, {0: "CT"},
                                  labels={
                                      "background": 0,
                                      "spleen": 1,
                                      "ignore": label_ignore
                                  },
                                  regions_class_order=(1, 2, 3),
                                  num_training_cases=num_training_cases,
                                  num_test_cases=num_test_cases,
                                  file_ending='.nii.gz',
                                  dataset_name=opts.dataset, reference='none',
                                  release='prerelease',
                                  overwrite_image_reader_writer='SimpleITKIO',
                                  description=opts.dataset)
        elif opts.dataset=='TAL':
            label_ignore = 2
            # Copy imagesTr
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTr') + '/*'))
            num_training_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTr'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    im.save(join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        fip_image_out = join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz')
                        sitk.WriteImage(imc, fip_image_out, True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy imagesTs
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTs') + '/*'))
            num_test_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTs'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    #sitk.WriteImage(im, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_0000.mhd'), True)
                    im.save(join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        #imc.SetDirection(im.GetDirection())
                        sitk.WriteImage(imc, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz'), True)
                else:
                    raise ValueError('The image dimension is not supported.')
            
            # Reform labels
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTr_raw') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTr_raw'):
                iml = sitk.ReadImage(fip_label)
                arr = sitk.GetArrayFromImage(iml)
                arr[arr==1] = 0
                arr[arr==6] = 1
                arr[arr==13] = 1
                arr[arr>1] = 0
                imln = sitk.GetImageFromArray(arr)
                imln.SetOrigin(iml.GetOrigin())
                imln.SetSpacing(iml.GetSpacing())
                imln.SetDirection(iml.GetDirection())
                sitk.WriteImage(imln, join(opts.fp_raw, opts.dataset, 'labelsTr', os.path.basename(fip_label).split('.')[0].replace('label','img')+'.nii.gz'), useCompression=True)

            
            # Copy labelTr
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTr') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTr'):
                #fip_image = join(imagestrTr, os.path.basename(fip_label).split('.')[0].replace('label','img') + '.nii.gz')
                fip_image = join(imagestrTr, os.path.basename(fip_label).split('.')[0].replace('label','img')+'_' + '0000' + '.nii.gz')
                iml = sitk.ReadImage(fip_label)
                im = sitk.ReadImage(fip_image)
                if delete_label:
                    arr = sitk.GetArrayFromImage(iml)
                    arr[:] = label_ignore
                    iml = sitk.GetImageFromArray(arr)
                    iml.SetOrigin(im.GetOrigin())
                    iml.SetSpacing(im.GetSpacing())
                    iml.SetDirection(im.GetDirection())
                else:
                    arr = sitk.GetArrayFromImage(iml)
                    iml = sitk.GetImageFromArray(arr)
                    iml.SetOrigin(im.GetOrigin())
                    iml.SetSpacing(im.GetSpacing())
                    iml.SetDirection(im.GetDirection())
                sitk.WriteImage(iml, join(labelstrTr, os.path.basename(fip_label).split('.')[0].replace('label','img')+'.nii.gz'), useCompression=True)
        
            # Copy labelTe
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTs') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTs'):
                im = sitk.ReadImage(fip_label)
                sitk.WriteImage(im, join(labelstrTs, os.path.basename(fip_label).split('.')[0]+'.nii.gz'), True)
        
            # Generate json file of dataset
            generate_dataset_json(out_base, {0: "CT"},
                                  labels={
                                      "background": 0,
                                      "organ": 1,
                                      "ignore": label_ignore
                                  },
                                  regions_class_order=(1, 2, 3),
                                  num_training_cases=num_training_cases,
                                  num_test_cases=num_test_cases,
                                  file_ending='.nii.gz',
                                  dataset_name=opts.dataset, reference='none',
                                  release='prerelease',
                                  overwrite_image_reader_writer='SimpleITKIO',
                                  description=opts.dataset)
        elif opts.dataset=='CTA':

            # Select datset
            label_ignore = 2
            # Copy imagesTr
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTr') + '/*.nrrd'))
            num_training_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTr'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    #im.save(join(imagestrTr, splitFilePath(fip_image)[1][0:-4]+'_0000.nii.gz'))
                    im.save(join(imagestrTr, splitFilePath(fip_image)[1]+'_0000.nrrd'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        fip_image_out = join(imagestrTr, splitFilePath(fip_image)[1]+'_' + nums + '.nii.gz')
                        sitk.WriteImage(imc, fip_image_out, True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy imagesTs
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTs') + '/*'))
            num_test_cases = len(fip_images)

            # Copy labelTr
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTr') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTr'):
                #fip_image = join(imagestrTr, os.path.basename(fip_label).split('.')[0].replace('label','img') + '.nii.gz')
                fip_image = join(imagestrTr, os.path.basename(fip_label).split('.')[0]+'_' + '0000' + '.nrrd')
                iml = sitk.ReadImage(fip_label)
                im = sitk.ReadImage(fip_image)
                if delete_label:
                    arr = sitk.GetArrayFromImage(iml)
                    arr[:] = label_ignore
                    iml = sitk.GetImageFromArray(arr)
                    iml.SetOrigin(im.GetOrigin())
                    iml.SetSpacing(im.GetSpacing())
                    iml.SetDirection(im.GetDirection())
                else:
                    arr = sitk.GetArrayFromImage(iml)
                    iml = sitk.GetImageFromArray(arr)
                    iml.SetOrigin(im.GetOrigin())
                    iml.SetSpacing(im.GetSpacing())
                    iml.SetDirection(im.GetDirection())
                sitk.WriteImage(iml, join(labelstrTr, os.path.basename(fip_label)), useCompression=True)
        
        elif (opts.dataset=='CTA17') or (opts.dataset=='CTA17C'):

            # Select datset
            label_ignore = 20
            labels={
                "background": 0,
                "pRCA": 1,
                "mRCA": 2,
                "dRCA": 3,
                "RPDA": 4,
                "RPLB": 16,
                "LM": 5,
                "pLAD": 6,
                "mLAD": 7,
                "dLAD": 8,
                "D1": 9,
                "D2": 10,
                "pLCX": 11,
                "OM1": 12,
                "mLCX": 13,
                "OM2": 14,
                "LPDA": 15,
                "LPLB": 18,
                "RAM": 17,
                "ND": 19,
                "ignore": label_ignore
            }

            # Copy imagesTr
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTr') + '/*.nii.gz'))
            num_training_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTr'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    #im.save(join(imagestrTr, splitFilePath(fip_image)[1][0:-4]+'_0000.nii.gz'))
                    im.save(join(imagestrTr, splitFilePath(fip_image)[1][0:-4]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        fip_image_out = join(imagestrTr, splitFilePath(fip_image)[1]+'_' + nums + '.nii.gz')
                        sitk.WriteImage(imc, fip_image_out, True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy imagesTs
            num_test_cases = 0

            # Create labelTr
            for fip_image in tqdm(fip_images, desc='Create labelTr'):
                im = sitk.ReadImage(fip_image)
                arr = sitk.GetArrayFromImage(im)
                arr[:] = label_ignore
                iml = sitk.GetImageFromArray(arr)
                iml.SetOrigin(im.GetOrigin())
                iml.SetSpacing(im.GetSpacing())
                iml.SetDirection(im.GetDirection())
                sitk.WriteImage(iml, join(labelstrTr, splitFilePath(fip_image)[1][0:-4]+'.nii.gz'), True)
   

            # Generate json file of dataset
            generate_dataset_json(out_base, {0: "CT"},
                                  labels=labels,
                                  regions_class_order=(1, 2, 3),
                                  num_training_cases=num_training_cases,
                                  num_test_cases=num_test_cases,
                                  file_ending='.nii.gz',
                                  dataset_name=opts.dataset, reference='none',
                                  release='prerelease',
                                  overwrite_image_reader_writer='SimpleITKIO',
                                  description=opts.dataset)
            
        elif opts.dataset=='CTA18':

            # Select datset
            label_ignore = 20
            labels={
                "background": 0,
                "pRCA": 1,
                "mRCA": 2,
                "dRCA": 3,
                "RPDA": 4,
                "RPLB": 16,
                "LM": 5,
                "pLAD": 6,
                "mLAD": 7,
                "dLAD": 8,
                "D1": 9,
                "D2": 10,
                "pLCX": 11,
                "OM1": 12,
                "mLCX": 13,
                "OM2": 14,
                "LPDA": 15,
                "LPLB": 18,
                "RAM": 17,
                "ND": 19,
                "ignore": label_ignore
            }

            # Copy imagesTr
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTr') + '/*.nii.gz'))
            num_training_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTr'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    #im.save(join(imagestrTr, splitFilePath(fip_image)[1][0:-4]+'_0000.nii.gz'))
                    im.save(join(imagestrTr, splitFilePath(fip_image)[1][0:-4]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        fip_image_out = join(imagestrTr, splitFilePath(fip_image)[1]+'_' + nums + '.nii.gz')
                        sitk.WriteImage(imc, fip_image_out, True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy imagesTs
            #fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTs') + '/*'))
            #num_test_cases = len(fip_images)num_test_cases
            num_test_cases = 0

            # Create labelTr
            for fip_image in tqdm(fip_images, desc='Create labelTr'):
                im = sitk.ReadImage(fip_image)
                arr = sitk.GetArrayFromImage(im)
                arr[:] = label_ignore
                iml = sitk.GetImageFromArray(arr)
                iml.SetOrigin(im.GetOrigin())
                iml.SetSpacing(im.GetSpacing())
                iml.SetDirection(im.GetDirection())
                sitk.WriteImage(iml, join(labelstrTr, splitFilePath(fip_image)[1][0:-4]+'.nii.gz'), True)
   
    

            # Generate json file of dataset
            generate_dataset_json(out_base, {0: "CT"},
                                  labels=labels,
                                  regions_class_order=(1, 2, 3),
                                  num_training_cases=num_training_cases,
                                  num_test_cases=num_test_cases,
                                  file_ending='.nii.gz',
                                  dataset_name=opts.dataset, reference='none',
                                  release='prerelease',
                                  overwrite_image_reader_writer='SimpleITKIO',
                                  description=opts.dataset)
            
        elif opts.dataset=='TCCT':

            # Select datset
            # label_ignore = 34
            # labels={
            #     "background": 0,
            #     "left_atrium": 1,
            #     "left_ventricle": 2,
            #     "left_ventricle_myocardium": 3,
            #     "right_atrium": 4,
            #     "right_ventricle": 5,
            #     "right_ventricle_myocardium": 6,
            #     "pulmonary_artery": 7,
            #     "descending_aorta": 8,
            #     "ascending_aorta": 9,
            #     "main_pulmonary": 10,
            #     "right_pulmonary_vein": 11,
            #     "left_pulmonary_vein": 12,
            #     "superior_vena_cava": 13,
            #     "epicardial_fat": 14,
            #     "pericardial_fat": 15,
            #     "lad": 16,
            #     "lcx": 17,
            #     "rca": 18,
            #     "coronary_sinus": 19,
            #     "aortic_valve": 20,
            #     "left_internal_mammary": 21,
            #     "right_internal_mammary": 22,
            #     "esophagus": 23,
            #     "left_interlob_pulmonary": 24,
            #     "left_mainstem_bronchus": 25,
            #     "left_lung": 26,
            #     "right_lung": 27,
            #     "left_atrial_appendage": 28,
            #     "sternum": 29,
            #     "rib": 30,
            #     "left_primary_bronchus": 31,
            #     "right_primary_bronchus": 32,
            #     "liver": 33,
            #     "ignore": label_ignore
            # }
            
            label_ignore = 37
            labels={
                "background": 0,
                "left_atrium": 1,
                "left_atrial_appendage": 2,
                "left_ventricle": 3,
                "left_ventricle_myocardium": 4,
                "right_atrium": 5,
                "right_ventricle": 6,
                "right_ventricle_myocardium": 7,
                "pulmonary_trunk": 8,
                "right_pulmonary_artery": 9,
                "left_pulmonary_artery": 10,
                "descending_aorta": 11,
                "ascending_aorta": 12,
                "right_pulmonary_veins": 13,
                "left_pulmonary_veins": 14,
                "superior_vena_cava": 15,
                "inferior_vena_cava": 16,
                "thoracic_fat": 17,
                "epicardial_fat": 18,
                "lm": 19,
                "lad": 20,
                "lcx": 21,
                "rca": 22,
                "coronary_sinus": 23,
                "aortic_valve": 24,
                "left_internal_mammary": 25,
                "right_internal_mammary": 26,
                "left_primary_bronchus": 27,
                "right_primary_bronchus": 28,
                "left_lung": 29,
                "right_lung": 30,
                "sternum": 31,
                "ribs": 32,
                "spine": 33,
                "esophagus": 34,
                "spleen": 35,
                "liver": 36,
                "ignore": label_ignore
            }
            
            # Copy imagesTr
            #fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTr') + '/*.nii.gz'))
            fip_images = sorted(glob(join(opts.fp_raw, 'CTA18', 'imagesTr') + '/*.nii.gz'))
            
            # Filter large fieled of view >250 mm
            fip_images = [x for x in fip_images if '13-RIG-0094' not in x]
            fip_images = [x for x in fip_images if '13-RIG-0169' not in x]
            
            num_training_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTr'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    #im.save(join(imagestrTr, splitFilePath(fip_image)[1][0:-4]+'_0000.nii.gz'))
                    im.save(join(imagestrTr, splitFilePath(fip_image)[1][0:-4]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        fip_image_out = join(imagestrTr, splitFilePath(fip_image)[1]+'_' + nums + '.nii.gz')
                        sitk.WriteImage(imc, fip_image_out, True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy imagesTs
            #fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTs') + '/*'))
            #num_test_cases = len(fip_images)num_test_cases
            num_test_cases = 0

            # Create labelTr
            for fip_image in tqdm(fip_images, desc='Create labelTr'):
                im = sitk.ReadImage(fip_image)
                arr = sitk.GetArrayFromImage(im)
                arr[:] = label_ignore
                iml = sitk.GetImageFromArray(arr)
                iml.SetOrigin(im.GetOrigin())
                iml.SetSpacing(im.GetSpacing())
                iml.SetDirection(im.GetDirection())
                sitk.WriteImage(iml, join(labelstrTr, splitFilePath(fip_image)[1][0:-4]+'.nii.gz'), True)
   
    

            # Generate json file of dataset
            generate_dataset_json(out_base, {0: "CT"},
                                  labels=labels,
                                  regions_class_order=(1, 2, 3),
                                  num_training_cases=num_training_cases,
                                  num_test_cases=num_test_cases,
                                  file_ending='.nii.gz',
                                  dataset_name=opts.dataset, reference='none',
                                  release='prerelease',
                                  overwrite_image_reader_writer='SimpleITKIO',
                                  description=opts.dataset)
            
        elif opts.dataset=='LIVG':
            label_ignore = 2
            # Copy imagesTr
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTr') + '/*'))
            num_training_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTr'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    im.save(join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        fip_image_out = join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz')
                        sitk.WriteImage(imc, fip_image_out, True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy imagesTs
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTs') + '/*'))
            num_test_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTs'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    #sitk.WriteImage(im, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_0000.mhd'), True)
                    im.save(join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        #imc.SetDirection(im.GetDirection())
                        sitk.WriteImage(imc, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz'), True)
                else:
                    raise ValueError('The image dimension is not supported.')
            
            # Reform labels
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTr_raw') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTr_raw'):
                iml = sitk.ReadImage(fip_label)
                arr = sitk.GetArrayFromImage(iml)
                arr[arr==1] = 0
                arr[arr==6] = 1
                arr[arr==12] = 1
                arr[arr==13] = 1
                arr[arr>1] = 0
                imln = sitk.GetImageFromArray(arr)
                imln.SetOrigin(iml.GetOrigin())
                imln.SetSpacing(iml.GetSpacing())
                imln.SetDirection(iml.GetDirection())
                sitk.WriteImage(imln, join(opts.fp_raw, opts.dataset, 'labelsTr', os.path.basename(fip_label).split('.')[0].replace('label','img')+'.nii.gz'), useCompression=True)

            
            # Copy labelTr
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTr') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTr'):
                #fip_image = join(imagestrTr, os.path.basename(fip_label).split('.')[0].replace('label','img') + '.nii.gz')
                fip_image = join(imagestrTr, os.path.basename(fip_label).split('.')[0].replace('label','img')+'_' + '0000' + '.nii.gz')
                iml = sitk.ReadImage(fip_label)
                im = sitk.ReadImage(fip_image)
                if delete_label:
                    arr = sitk.GetArrayFromImage(iml)
                    arr[:] = label_ignore
                    iml = sitk.GetImageFromArray(arr)
                    iml.SetOrigin(im.GetOrigin())
                    iml.SetSpacing(im.GetSpacing())
                    iml.SetDirection(im.GetDirection())
                else:
                    arr = sitk.GetArrayFromImage(iml)
                    iml = sitk.GetImageFromArray(arr)
                    iml.SetOrigin(im.GetOrigin())
                    iml.SetSpacing(im.GetSpacing())
                    iml.SetDirection(im.GetDirection())
                sitk.WriteImage(iml, join(labelstrTr, os.path.basename(fip_label).split('.')[0].replace('label','img')+'.nii.gz'), useCompression=True)
        
            # Copy labelTe
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTs') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTs'):
                im = sitk.ReadImage(fip_label)
                sitk.WriteImage(im, join(labelstrTs, os.path.basename(fip_label).split('.')[0]+'.nii.gz'), True)
        
            # Generate json file of dataset
            generate_dataset_json(out_base, {0: "CT"},
                                  labels={
                                      "background": 0,
                                      "organ": 1,
                                      "ignore": label_ignore
                                  },
                                  regions_class_order=(1, 2, 3),
                                  num_training_cases=num_training_cases,
                                  num_test_cases=num_test_cases,
                                  file_ending='.nii.gz',
                                  dataset_name=opts.dataset, reference='none',
                                  release='prerelease',
                                  overwrite_image_reader_writer='SimpleITKIO',
                                  description=opts.dataset)
        elif opts.dataset=='LIVG2':
            label_ignore = 2
            # Copy imagesTr
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTr') + '/*'))
            num_training_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTr'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    im.save(join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        fip_image_out = join(imagestrTr, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz')
                        sitk.WriteImage(imc, fip_image_out, True)
                else:
                    raise ValueError('The image dimension is not supported.')
                    
            # Copy imagesTs
            fip_images = sorted(glob(join(opts.fp_raw, opts.dataset, 'imagesTs') + '/*'))
            num_test_cases = len(fip_images)
            for fip_image in tqdm(fip_images, desc='Copy imagesTs'):
                im = CTImage(fip_image)
                arr = im.image()
                if len(arr.shape)==3:
                    #sitk.WriteImage(im, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_0000.mhd'), True)
                    im.save(join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_0000.nii.gz'))
                elif len(arr.shape)==4:
                    for c in range(arr.shape[0]):
                        nums = str(c).zfill(4)
                        imc = sitk.GetImageFromArray(arr[c])
                        imc.SetOrigin(im.GetOrigin())
                        imc.SetSpacing(im.GetSpacing())
                        #imc.SetDirection(im.GetDirection())
                        sitk.WriteImage(imc, join(imagestrTs, os.path.basename(fip_image).split('.')[0]+'_' + nums + '.nii.gz'), True)
                else:
                    raise ValueError('The image dimension is not supported.')
            
            # Reform labels
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTr_raw') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTr_raw'):
                iml = sitk.ReadImage(fip_label)
                arr = sitk.GetArrayFromImage(iml)
                arr[arr==1] = 0
                arr[arr==6] = 1
                #arr[arr==12] = 1
                arr[arr==13] = 1
                arr[arr>1] = 0
                imln = sitk.GetImageFromArray(arr)
                imln.SetOrigin(iml.GetOrigin())
                imln.SetSpacing(iml.GetSpacing())
                imln.SetDirection(iml.GetDirection())
                sitk.WriteImage(imln, join(opts.fp_raw, opts.dataset, 'labelsTr', os.path.basename(fip_label).split('.')[0].replace('label','img')+'.nii.gz'), useCompression=True)
        
            
            # Copy labelTr
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTr') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTr'):
                #fip_image = join(imagestrTr, os.path.basename(fip_label).split('.')[0].replace('label','img') + '.nii.gz')
                fip_image = join(imagestrTr, os.path.basename(fip_label).split('.')[0].replace('label','img')+'_' + '0000' + '.nii.gz')
                iml = sitk.ReadImage(fip_label)
                im = sitk.ReadImage(fip_image)
                if delete_label:
                    arr = sitk.GetArrayFromImage(iml)
                    arr[:] = label_ignore
                    iml = sitk.GetImageFromArray(arr)
                    iml.SetOrigin(im.GetOrigin())
                    iml.SetSpacing(im.GetSpacing())
                    iml.SetDirection(im.GetDirection())
                else:
                    arr = sitk.GetArrayFromImage(iml)
                    iml = sitk.GetImageFromArray(arr)
                    iml.SetOrigin(im.GetOrigin())
                    iml.SetSpacing(im.GetSpacing())
                    iml.SetDirection(im.GetDirection())
                sitk.WriteImage(iml, join(labelstrTr, os.path.basename(fip_label).split('.')[0].replace('label','img')+'.nii.gz'), useCompression=True)
        
            # Copy labelTe
            fip_labels = sorted(glob(join(opts.fp_raw, opts.dataset, 'labelsTs') + '/*'))
            for fip_label in tqdm(fip_labels, desc='Copy labelsTs'):
                im = sitk.ReadImage(fip_label)
                sitk.WriteImage(im, join(labelstrTs, os.path.basename(fip_label).split('.')[0]+'.nii.gz'), True)
        
            # Generate json file of dataset
            generate_dataset_json(out_base, {0: "CT"},
                                  labels={
                                      "background": 0,
                                      "organ": 1,
                                      "ignore": label_ignore
                                  },
                                  regions_class_order=(1, 2, 3),
                                  num_training_cases=num_training_cases,
                                  num_test_cases=num_test_cases,
                                  file_ending='.nii.gz',
                                  dataset_name=opts.dataset, reference='none',
                                  release='prerelease',
                                  overwrite_image_reader_writer='SimpleITKIO',
                                  description=opts.dataset)
        else:
            raise ValueError('Dataset conversation not implemented for dataset: ' + opts.dataset)
            
    
    def load_data(self, opts, folderDict, man, data, batch_size=8, create_pseudo=True, create_mask=True, previous=True, device='cuda'):
        # data=man.datasets['train'].data
        # Create dataloader
        load_weights=create_pseudo
        net = man.load_model(opts, folderDict, previous=previous, load_weights=load_weights)
        dataloader_train = net.model['unet'].get_dataloaders_alunet(data_load=data, batch_size=batch_size)
        NumBatches = math.ceil(len(data)/dataloader_train.data_loader.batch_size)
        
        NumClasses = len(load_json(os.path.join(nnUNet_raw, opts.DName, 'dataset.json'))['labels'])-1
        for b in tqdm(range(NumBatches), desc='Load patch'):
            batch = next(dataloader_train)
            datab = batch['data']
            IDX = batch['idx'][:,0]
            imagenames = batch['imagename']
            datab = datab.to(device, non_blocking=True)
            if create_pseudo:
                out = net.model['unet'].network(datab)
                for i in range(len(out)): out[i] = out[i].detach_().cpu()
                pred=soft1(out[0])
            
            # Set prediction and mask 
            fp_full_labels = join(nnUNet_preprocessed, opts.DNameFull, opts.plans_identifier+'_'+opts.configuration)
            for i,ID in enumerate(list(IDX)):
                if create_mask:
                    if i==0 or imagenames[i,0]!=imagenames[i-1,0]:
                        fip = glob(join(fp_full_labels, imagenames[i,0]+'.npz'))[0]
                        ref = CTRef(np.load(fip)['seg'][0])
                        mask = ref.numTobin(NumClasses).ref()
                if create_pseudo:
                    data[ID].P['XMaskPred'] = compute_one_hot_torch(pred[i:i+1])
                data[ID].X['XImage'] = datab[i:i+1]
                if create_mask:
                    data[ID].Y['XMask'] = dataloader_train.data_loader.generate_mask(data[ID], mask)
                    
            
    def query(self, opts, folderDict, man, NumSamples, batchsize=2, NumSamplesMaxMCD=5000):
        previous=True 
        strategy = strategies[opts.strategy]()
        self.man.alunet = self
        #action_round = strategy.query(opts, folderDict, man, man.datasets['query'].data, opts.CLPatch, NumSamples=NumSamples, batchsize=500, pred_class='XMaskPred', previous=previous, save_uc=False)
        action_round = strategy.query(opts, folderDict, man, man.datasets['query'].data, opts.CLPatch, NumSamples=NumSamples, batchsize=batchsize, pred_class='XMaskPred', previous=previous, save_uc=False, NumSamplesMaxMCD=NumSamplesMaxMCD)
        man.datasets['action_round'].data = action_round
        man.save(include=['action_round', 'train','query'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
    
    def select_manual(self, opts, folderDict, man, data_action, NumSamples):
        # self=alunet
        # man=self.man
        
        if 'USIMFT' in opts.strategy:
            strategy_name  = 'USIMFT'
        else:
            strategy_name = opts.strategy
        previous=True 
        strategy = strategies[strategy_name]()
        man.alunet = self
        data_select = strategy.select_manual2(opts, folderDict, man, man.datasets['query'].data, data_action, NumSamples)
        #man.save(include=['action_round', 'train','query'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
        return data_select
        
    def annotation_auto(self, opts, man, data, full=False, update_train=True):
        # data=man.datasets['valid'].data
        # self.load_data(opts, folderDict, man, data, batch_size=8)
        dataSort = UNetPatchV2.sort_F(data, prop='imagename')
        imagenames = sorted(np.unique([s.F['imagename'] for s in data]))
        fp_labelsTr = os.path.join(join(nnUNet_raw, opts.DName), 'labelsTr')
        file_ending = load_json(join(nnUNet_preprocessed, opts.DName, 'dataset.json'))['file_ending']
        for imn in tqdm(imagenames, desc='Auto annotation'):
            fip = os.path.join(fp_labelsTr, imn + file_ending)
            fip_raw = join(opts.fp_raw, opts.dataset, 'labelsTr', imn + file_ending)
            ref = CTRef(fip)
            arr = ref.ref()
            arr_raw = CTRef(fip_raw).ref()
            if full:
                arr = arr_raw
            else:
                for s in dataSort:
                    if s.F['imagename']==imn:
                        lbs_org = s.F['lbs_org']
                        ubs_org = s.F['ubs_org']
                        arr[lbs_org[0]:ubs_org[0], lbs_org[1]:ubs_org[1], lbs_org[2]:ubs_org[2]] = arr_raw[lbs_org[0]:ubs_org[0], lbs_org[1]:ubs_org[1], lbs_org[2]:ubs_org[2]]
            ref.setRef(arr)
            ref.save(fip)
        
        # Update training set
        if update_train:
            man.datasets['train'].data = man.datasets['train'].data + man.datasets['action_round'].data
            man.datasets['action_round'].data=[]
            man.save(include=['train','action_round'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
        return True

    def annotation_auto_correction(self, opts, man, data, full=False):
        # data=man.datasets['valid'].data
        # self.load_data(opts, folderDict, man, data, batch_size=8)
        dataSort = UNetPatchV2.sort_F(data, prop='imagename')
        imagenames = sorted(np.unique([s.F['imagename'] for s in data]))
        fp_labelsTr = os.path.join(join(nnUNet_raw, opts.DName), 'labelsTr')
        file_ending = load_json(join(nnUNet_preprocessed, opts.DName, 'dataset.json'))['file_ending']
        for imn in tqdm(imagenames, desc='Auto annotation'):
            fip = os.path.join(fp_labelsTr, imn + file_ending)
            fip_raw = join(opts.fp_raw, opts.dataset, 'labelsTr', imn + file_ending)
            ref = CTRef(fip)
            arr = ref.ref()
            arr_raw = CTRef(fip_raw).ref()
            if full:
                arr = arr_raw
            else:
                for s in dataSort:
                    if s.F['imagename']==imn:
                        lbs_org = s.F['lbs_org']
                        ubs_org = s.F['ubs_org']
                        arr[lbs_org[0]:ubs_org[0], lbs_org[1]:ubs_org[1], lbs_org[2]:ubs_org[2]] = arr_raw[lbs_org[0]:ubs_org[0], lbs_org[1]:ubs_org[1], lbs_org[2]:ubs_org[2]]
            ref.setRef(arr)
            ref.save(fip)
        
        # Update training set
        #man.datasets['train'].data = man.datasets['train'].data + man.datasets['action_round'].data
        #man.datasets['action_round'].data=[]
        man.save(include=['train','action_round'], save_dict={}, save_class=opts.CLPatch, hdf5=False)


    def create_action(self, opts, man, data, filetype='.nii.gz', classification=False, classification_multislice=False, show_roi=True, create_pseudo=True, create_mask=True, pseude_is_mask=False, info_classified=False, info_selected=False, info_annotated=False, pseudo_full=True):
        # data=self.man.datasets['action_pre'].data
        # man=self.man
        # create_pseudo=False
        # create_mask=False
        # show_roi=True
        # classification=True
        # classification_multislice=True
        # pseudo_full=True
        # filetype='.nii.gz'
        
        
        folderDict = self.man.folderDict
    
       
        # Create folder structure
        if opts.fp_manual is None:
            fp_manual = join(os.path.dirname(man.folderDict['modelpath']), 'data_manual')
        else:
            fp_manual = opts.fp_manual
            
        # Delete fp_manual
        if os.path.isdir(fp_manual):
            shutil.rmtree(fp_manual, ignore_errors=True)
        os.makedirs(fp_manual, exist_ok=True)

        foldername = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
        out_base = os.path.join(nnUNet_raw, foldername)
        labelstrTr = os.path.join(out_base, "labelsTr")
            
        fp_images = os.path.join(fp_manual, 'images')
        fp_pseudo = os.path.join(fp_manual, 'pseudo')
        fp_refine = os.path.join(fp_manual, 'refine')
        fp_mask = os.path.join(fp_manual, 'mask')
        fip_actionlist = os.path.join(fp_manual, 'actionlist.json')
        fip_color = os.path.join(fp_manual, 'XALabelerLUT.ctbl')
        os.makedirs(fp_pseudo, exist_ok=True)
        os.makedirs(fp_refine, exist_ok=True)
        os.makedirs(fp_mask, exist_ok=True)
        
        labels = load_json(os.path.join(nnUNet_preprocessed, opts.DName, 'dataset.json'))['labels']
        
        # Create colors
        colors_raw=[]
        if len(labels)>24:
            colors_raw_arr = np.round(np.random.uniform(low=0.0, high=1.0, size=(len(labels)+3,3)), decimals=1)
            for i in range(colors_raw_arr.shape[0]):
                colors_raw.append([colors_raw_arr[i,0], colors_raw_arr[i,1], colors_raw_arr[i,2]])
        else:
            colors_raw = [[0,0,0], [1,0,0], [0,0,1], [1,1,0], 
                          [0,1,1], [1,0,1], [0.75,0.5,0],[0.5,0.5,0.5],
                          [0.5,0,0], [0.5,0.5,0], [0,0.5,0], [0.5,0,0.5],
                          [0,0.5,0.5], [0,0,0.5], [1,1,1], [0.9,0.5,0.2], [0.2,0.5,0.2], [0.2,0.5,0.9],
                          [0.2,0.2,0.2], [0.5,0.5,0.2], [0.3,0.4,0.2], [0.7,0.2,0.8], [0.2,0.7,0.8],
                          [0.7,0.2,0.8]]
        
        
        

        if classification:
            labels['positive']=labels['ignore']+1
            labels['negative']=labels['ignore']+2
            labels['pseudo']=labels['ignore']+3
        else:
            labels['pseudo']=labels['ignore']+1
            
        colors=[]
        # for i in range(len(labels)-1):
        #     key = list(labels.keys())[list(labels.values()).index(i)]
        #     colors.append([i, key, colors_raw[i]])
        for i in range(len(labels)):
            key = list(labels.keys())[list(labels.values()).index(i)]
            if key =='ignore':
                colors.append([i, key, [0,1,0]])
            else:
                colors.append([i, key, colors_raw[i]])                
        lines = ['# Color']
        for i in range(len(colors)):
            if colors[i][1]=='ignore':
                trans = '0'
            else:
                trans = '255'
            col = str(colors[i][0]) + ' ' + str(colors[i][1]) + ' ' + str(colors[i][2][0]*255) + ' ' + str(colors[i][2][1]*255) + ' ' + str(colors[i][2][2]*255) + ' ' + trans
            lines = lines + [col]
        with open(fip_color, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n') 

        # Create settings file
        settings={'method': 'xalabeler',
                  'fp_images': os.path.join(nnUNet_raw, opts.DName, 'imagesTr'),
                  'fp_pseudo': fp_pseudo,
                  'fp_refine': fp_refine,
                  'fp_mask': fp_mask,
                  'fip_actionlist': fip_actionlist,
                  'fip_colors': fip_color,
                  'foregroundOpacity': 0.3,
                  "classification": classification,
                  "classification_multislice": classification_multislice,
                  "show_roi": show_roi,
                  "fip_round_status": man.folderDict['fip_round_status'],
                  "default_ignore": False}
        
        fip_settings = os.path.join(fp_manual, 'settings_XALabeler.json')
        with open(fip_settings, 'w') as file:
            file.write(json.dumps(settings, indent=4))
            
        # Create pseudo label
        # if not pseudo_full:
        #     self.load_data(opts, folderDict, man, data, batch_size=8, create_pseudo=create_pseudo, create_mask=create_mask)
        

                
            
        # Create action list
        fp_nnunetData = os.path.join(nnUNet_raw, opts.DName)
        fp_imagesTr = os.path.join(fp_nnunetData, 'imagesTr')
        
        # Sort by imagename
        dataSort = UNetPatchV2.sort_F(data, prop='imagename', prop_sorted=False)

        # Create actionlist
        actionlist=[]
        for s in dataSort:
            action=ALAction()
            action.name='action'
            action.status='open'
            action.id=s.F['ID']
            action.bboxLbsOrg=[x for x in list(s.F['lbs_org'])]
            action.bboxUbsOrg=[x for x in list(s.F['ubs_org'])]
            action.dim=opts.dim
            action.imagename=s.F['imagename']+'_0000' + filetype
            action.pseudoname=s.F['imagename'] + filetype
            action.refinename=s.F['imagename'] + filetype
            action.filetype=filetype
            action.maskname=None
            action.label=colors
            action.info['class']=[]
            action.info['classified']=info_classified
            #action.info['selected']=info_selected
            action.info['annotated']=info_annotated
            actionlist.append(action)
        ALAction.save(fip_actionlist, actionlist)
        
        if create_pseudo:
            if not pseudo_full:
                self.load_data(opts, folderDict, man, data, batch_size=8, create_pseudo=create_pseudo, create_mask=create_mask)
            opts.CLSample.savePseudo(opts, data, folderDict, fp_pseudo, filetype, pseudo_full=pseudo_full, tile_step_size=0.7)
            #self.predict_pseudo_from_files(opts, data, folderDict, fp_pseudo, filetype, pseudo_full=True, tile_step_size=0.9)
            
        
        if pseude_is_mask:
            shutil.rmtree(fp_pseudo)
            shutil.copytree(labelstrTr, fp_pseudo, dirs_exist_ok=True)
            #fip_labels_mask = glob(labelstrTr + '/*')
            #for fip_label in fip_labels_mask:
        
    def predict_pseudo_from_files(self, opts, data, folderDict, fp_pseudo, filetype, pseudo_full=False, tile_step_size=0.5):
        
        dataSort = UNetPatchV2.sort_F(data, prop='imagename')
        
        #plans = load_json(join(nnUNet_preprocessed, opts.DName, 'nnUNetPlans.json'))
        #plans_manager = PlansManager(plans)
        #configuration_manager = plans_manager.get_configuration(opts.configuration)
        
        #idx = np.argsort([s.F['imagename'] for s in dataSort])
        fp_nnunetData = os.path.join(nnUNet_raw, opts.DName)
        fp_imagesTr = os.path.join(fp_nnunetData, 'imagesTr')
        #fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')
        #imagenames = list(np.sort([s.F['imagename'] for s in dataSort]))
        imagenames = list([s.F['imagename'] for s in dataSort])
        
        imagenames = list(np.unique(imagenames))
        fip_images=[]
        fip_pseudos=[]
        for i,imn in enumerate(tqdm(imagenames, desc='Save pseudo label')):
            fip_image = glob(fp_imagesTr + '/'+imn+'_*')[0]
            print('fip_image123', fip_image)
            fip_pseudo = os.path.join(fp_pseudo, imn + filetype)
            modelpath = opts.alunet.man.folderDict['modelpath_prev']
            #opts.alunet.predict_image(opts, folderDict, modelpath, fip_image, fip_pseudo)
            fip_images.append(fip_image)
            fip_pseudos.append(fip_pseudo)
        
        # Create predictor
        predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_gpu=True,
            device=torch.device('cuda', 0),
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True
            )
        
        predictor.initialize_from_trained_model_folder(
            os.path.join(modelpath, opts.nnUNetResults),
            use_folds=(opts.fold, ),
            checkpoint_name='checkpoint_final.pth',
        )

        # Predict image
        #image = CTImage(fip_image)
        #img, props = SimpleITKIO().read_images([fip_image])
        print('fp_imagesTr123', fp_imagesTr)
        print('fp_pseudo123', fp_pseudo)
        ret = predictor.predict_from_files(fp_imagesTr, fp_pseudo)
        
        
        
    def predict_image(self, opts, folderDict, modelpath, fip_image, fip_predict, tile_step_size=0.5):
        
        # self=alunet
        # fp_images='
        
        #NewVersion = False
        #VersionUse = None
        #strategy_name = opts.strategy
        #man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        #folderDict = self.folderDict = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=True)
        #self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)

        # Create predictor
        predictor = nnUNetPredictor(
            #tile_step_size=0.5,
            tile_step_size=tile_step_size,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_gpu=True,
            device=torch.device('cuda', 0),
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True
            )
        
        predictor.initialize_from_trained_model_folder(
            os.path.join(modelpath, opts.nnUNetResults),
            use_folds=(opts.fold, ),
            checkpoint_name='checkpoint_final.pth',
        )

        # Predict image
        image = CTImage(fip_image)
        img, props = SimpleITKIO().read_images([fip_image])
        ret = predictor.predict_single_npy_array(img, props, None, None, False)
        pred = CTRef(ret)
        pred.copyInformationFrom(image)
        pred.save(fip_predict)
        del image
        del pred
        
    

    def annotation_manual(self, opts, man, data_action, bg_offset=False, ignore_pseudo=False):
        
        # data_action=self.man.datasets['action_round'].data
        # man=alunet.man
        # bg_offset=False
        
        if len(data_action)==0:
            return None
        
        if opts.fp_manual is None:
            fp_manual = join(os.path.dirname(man.folderDict['modelpath']), 'data_manual')
        else:
            fp_manual = opts.fp_manual
            
        #print('fp_manual123', fp_manual)
            
        fip_actionlist = os.path.join(fp_manual, 'actionlist.json')
        # !!! in load function
        if not os.path.isfile(fip_actionlist):
            sys.exit('Could not find actionlist.json.')
        actionlist = ALAction.load(fip_actionlist) 
        fp_nnunetData = os.path.join(nnUNet_raw, opts.DName)
        fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')
        
        labels = load_json(os.path.join(nnUNet_preprocessed, opts.DName, 'dataset.json'))['labels']
        label_ignore = labels['ignore']
        
        #print('actionlist123', actionlist)
        if ignore_pseudo:
            label_offset = labels['ignore']
        else:
            label_offset = 0
            
        # !!!
        #for s in data_action:
        #    print('IDS', s.F['ID'])
        #sys.exit('TEST2')

        #fip_refine = None
        annotated=True
        pbar = tqdm(total=len(data_action))
        pbar.set_description("Update action" )
        for a in actionlist:
            print('a123', a)
            # !!!
            #if not hasattr(a, 'filetype'):
            a.filetype='.nii.gz'
            
            pbar.update()
            if a.status=='solved':
                #sys.exit()
                fip_refine = join(fp_manual, 'refine', a.pseudoname)
                s = opts.CLSample.getSampleByID(data_action, a.id)
                ref_refine = CTRef(fip_refine)
                arrR = ref_refine.ref()
                if bg_offset:
                    arrR = arrR-1
                    arrR[arrR==-1]=label_offset
                fip_label = os.path.join(fp_labelsTr, s.F['imagename']+a.filetype)
                ref_label = CTRef(fip_label)
                arrL = ref_label.ref()
                #arrLT = arrL[s.F['lbs_org'][0]:s.F['ubs_org'][0], s.F['lbs_org'][1]:s.F['ubs_org'][1], s.F['lbs_org'][2]:s.F['ubs_org'][2]]
                #arrRT = arrR[s.F['lbs_org'][0]:s.F['ubs_org'][0], s.F['lbs_org'][1]:s.F['ubs_org'][1], s.F['lbs_org'][2]:s.F['ubs_org'][2]]
                #arrBin = (arrRT!=label_ignore)*1
                #arrLT = (1-arrBin) * arrLT + arrBin * arrRT
                #arrL[s.F['lbs_org'][0]:s.F['ubs_org'][0], s.F['lbs_org'][1]:s.F['ubs_org'][1], s.F['lbs_org'][2]:s.F['ubs_org'][2]] = arrLT
                
                arrBin = (arrR!=label_ignore)*1
                arrL = (1-arrBin) * arrL + arrBin * arrR
                
                
                ref_label.setRef(arrL)
                ref_label.save(fip_label)
                print('fip_label123', fip_label)
            else:
                annotated=False
        pbar.close()
        
        # Update training set
        if annotated:
            man.datasets['train'].data = man.datasets['train'].data + man.datasets['action_round'].data
            man.datasets['action_round'].data=[]
            man.save(include=['train','action_round'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
            
        #sys.exit()
        
        return annotated
        
    def train(self, opts, copy_model=True, preprocess=False):
              
        # Load current round
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=False)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        
        # Preprocessing raw data
        if preprocess:
            #subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            #subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -overwrite_plans_name ALUNETPlanner" +  " -pl ALUNETPlanner"  + " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -overwrite_plans_name " + opts.plans_identifier +  " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
        
        run_training(dataset_name_or_id=opts.dataset_name_or_id, 
                      configuration=opts.configuration, 
                      fold=opts.fold,
                      trainer_class_name=opts.nnUNetTrainer, 
                      plans_identifier=opts.plans_identifier, 
                      pretrained_weights=None,
                      num_gpus=1, 
                      use_compressed_data=False, 
                      export_validation_probabilities=False, 
                      continue_training=False, 
                      only_run_validation=False, 
                      disable_checkpointing=False, 
                      val_with_best=False,
                      device=torch.device('cuda'))   

        # Copy model and labeles
        if copy_model:
            self.copy_model_labels(opts, self.man.folderDict)
            
    def trainval(self, opts, copy_model=True, modelpath='modelpath_prev'):
        
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=True)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        
        run_training(dataset_name_or_id=opts.dataset_name_or_id, 
                     configuration=opts.configuration, 
                     fold=opts.fold,
                     trainer_class_name=opts.nnUNetTrainer, 
                     plans_identifier=opts.plans_identifier, 
                     pretrained_weights=os.path.join(folderDict[modelpath], opts.nnUNetResults,'fold_'+str(opts.fold), 'checkpoint_final.pth'),
                     num_gpus=1, 
                     use_compressed_data=False, 
                     export_validation_probabilities=True, 
                     continue_training=False, 
                     only_run_validation=False, 
                     disable_checkpointing=False, 
                     val_with_best=True,
                     device=torch.device('cuda'))  
        
        # Copy model and labeles
        if copy_model:
            self.copy_model_labels(opts, self.man.folderDict)

    def test(self, opts, modelpath='modelpath_prev'):
        
        # self=alunet
        
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=True)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        
        # Predict train images
        fp_test = join(os.path.dirname(folderDict['fp_manager']), 'data', 'test')
        os.makedirs(fp_test, exist_ok=True)
        #modelpath='modelpath'
        #fip_image = '/mnt/HHD/data/UNetAL/data_raw/CTA17C/imagesTr/05-COP-0031_1.2.392.200036.9116.2.2424156352.1455761338.8.1264100003.1.nii.gz'
        #fip_predict = '/mnt/HHD/data/UNetAL/nnunet/nnUNet_results/Dataset604_CTA17C/nnUNetTrainer_ALUNET__nnUNetPlans__2d/fold_0/validation/image.nii.gz'
        modelpath = self.man.folderDict['modelpath_prev']
        fip_images = glob(join(nnUNet_raw, opts.DName, 'imagesTr/*'))
        
        # Filter masks
        fp_masks = '/mnt/HHD/data/CTAAutoplaque/data/surenjav/CADMAN/export/masks'
        fp_masks_pat = glob(fp_masks + '/*')
        
        patients = [os.path.basename(fip).split('_')[0] for fip in fip_images if os.path.basename(fip).split('_')[0] in [os.path.basename(fipm) for fipm in fp_masks_pat]]
        fip_images = sorted([fip for fip in fip_images if os.path.basename(fip).split('_')[0] in patients])
        fp_masks = sorted([fip for fip in fp_masks_pat if os.path.basename(fip) in patients])
        NumClasses=19
        
        fip_export = join(fp_test, 'perform.pkl')

        perform={'images': []}
        for fip_image, fp_mask in tqdm(zip(fip_images, fp_masks), desc='Evaluate prediction'):
            fip_predict = join(fp_test, os.path.basename(fip_image))
            fip_mask = join(fp_mask, 'segment_masks.mhd')
            self.predict_image(opts, folderDict, modelpath, fip_image, fip_predict)
            
            ref_pred = CTRef(fip_predict)
            ref_mask = CTRef(fip_mask)
            pred = ref_pred.numTobin(NumClasses, offset=0).ref().swapaxes(1, 3).reshape(-1, NumClasses).astype(np.int16)
            mask = ref_mask.numTobin(NumClasses, offset=0).ref().swapaxes(1, 3).reshape(-1, NumClasses).astype(np.int16)
            #ref_mask = CTRef(fip_mask).
            
            # Compute performance per image
            C = confusion(torch.from_numpy(mask), torch.from_numpy(pred), num_classes=NumClasses, device='cpu').numpy()
            p=dict()
            p['C'] = C
            p['F1'] = f1(C)
            p['F1_micro'] = f1(C[1:,1:], mode='micro')
            p['F1_bin'] = f1(C, binary_class=None)
            
            # Compute overall performance
            perform['images'].append(p)
            if 'C' not in perform:
                perform['C'] = C
            else:
                perform['C'] = perform['C'] + C
            perform['F1'] = f1(perform['C'])
            perform['F1_micro'] = f1(perform['C'][1:,1:], mode='micro')
            perform['F1_bin'] = f1(perform['C'], binary_class=None)
            
            # Save performance
            pickle.dump(perform, open(fip_export, 'wb'))
            
        # Read performance
        with open(fip_export, 'rb') as f:
            perform = pickle.load(f)

        
    def val(self, opts, copy_model=True, modelpath='modelpath_prev', val_with_best=False):

        
        # self=alunet
        
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=True)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        
        run_training(dataset_name_or_id=opts.dataset_name_or_id, 
                     configuration=opts.configuration, 
                     fold=opts.fold,
                     trainer_class_name=opts.nnUNetTrainer, 
                     plans_identifier=opts.plans_identifier, 
                     pretrained_weights=os.path.join(folderDict[modelpath], opts.nnUNetResults,'fold_'+str(opts.fold), 'checkpoint_final.pth'),
                     num_gpus=1, 
                     use_compressed_data=False, 
                     #export_validation_probabilities=True, 
                     export_validation_probabilities=False, 
                     continue_training=False, 
                     only_run_validation=True, 
                     disable_checkpointing=False, 
                     val_with_best=val_with_best,
                     device=torch.device('cuda'))  
        
        # Copy model and labeles
        if copy_model:
            self.copy_model_labels(opts, self.man.folderDict)
            
    def train_step(self, opts, copy_model=True, modelpath='modelpath_prev', pretrained=True):
        
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=True)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        if pretrained:
            pretrained_weights = os.path.join(folderDict['modelpath_prev'], opts.nnUNetResults,'fold_'+str(opts.fold), 'checkpoint_final.pth')
        else:
            pretrained_weights = None
            
        #print('pretrained_weights123', pretrained_weights)
        #sys.exit()
        
        run_training(dataset_name_or_id=opts.dataset_name_or_id, 
                     configuration=opts.configuration, 
                     fold=opts.fold, 
                     trainer_class_name=opts.nnUNetTrainer, 
                     plans_identifier=opts.plans_identifier, 
                     pretrained_weights=pretrained_weights,
                     #pretrained_weights=None,
                     num_gpus=1, 
                     use_compressed_data=False, 
                     export_validation_probabilities=False, 
                     continue_training=False, 
                     #continue_training=True, 
                     only_run_validation=False, 
                     disable_checkpointing=False, 
                     val_with_best=False,
                 device=torch.device('cuda'))
        
        # Copy model and labeles
        if copy_model:
            self.copy_model_labels(opts, self.man.folderDict)
        
    def alload(self, opts, datatset='train'):
        # self=alunet
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=False)
        man.load(include=['train'], load_class=opts.CLPatch, hdf5=False)
        self.load_data(opts, folderDict, man, data=man.datasets[datatset].data)
        
        # for s in man.datasets['train'].data:
        #     s.plotSample(plotlist=['XImage', 'P', 'Y'], color=True)
        
    def alcreate(self, opts, copy_nnUNet_data=False):
        NewVersion = True
        VersionUse = None
        strategy_name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        self.man.folderDict = man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=copy_nnUNet_data)
        man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        for key in self.man.folderDict['round_status']: self.man.folderDict['round_status'][key]=False
        return man, self.man.folderDict

    def alquerytest(self, opts):
        
        # opts.label_manual = True
        # opts.strategy = 'RANDOM'
        # opts.ALSamples[folderDict['version']]=100
        # self=alunet
        
        # Load current round
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = self.folderDict = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=True)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        
        previous=True 
        strategy = strategies[opts.strategy]()
        man.alunet = self
        #action_round = strategy.query(opts, folderDict, man, man.datasets['query'].data, opts.CLPatch, NumSamples=NumSamples, batchsize=500, pred_class='XMaskPred', previous=previous, save_uc=False)

    def alround(self, opts, copy_nnUNet_data=True):
        if opts.targeted:
            self.alround_target(opts, copy_nnUNet_data)
        else:
            self.alround_plain(opts, copy_nnUNet_data)
            
    
    def alround_plain(self, opts, copy_nnUNet_data=True):
        
        # opts.label_manual = True
        # opts.strategy = 'RANDOM'
        # opts.ALSamples[folderDict['version']]=100
        # self=alunet
        # copy_nnUNet_data=False
        
        # Load current round
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        _ = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        _= self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=copy_nnUNet_data)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        opts.alunet = self
        
        # !!!
        #self.man.datasets['action_round'].data = self.man.datasets['train'].data[100:]
        #self.man.datasets['train'].data = [] self.man.datasets['train'].data[0:100]
        #self.man.info()
        #sys.exit()
        
        #print('info123', self.man.info())
        #sys.exit()
        

        if opts.label_manual:
            processed = np.all([self.man.folderDict['round_status'][key] for key in self.man.folderDict['round_status']])
            if processed:
                # Create new AL round  
                _, _ = self.alcreate(opts, copy_nnUNet_data=True)
                
            # Correction
            if opts.correction:
                pass
            else:
                self.man.update_status(self.man.folderDict, 'correction_proposal', True)
                self.man.update_status(self.man.folderDict, 'correction_selection_manual', True)
                self.man.update_status(self.man.folderDict, 'correction_preparation', True)
                self.man.update_status(self.man.folderDict, 'correction_annotation_manual', True)
            
            # Set True
            self.man.update_status(self.man.folderDict, 'selection_proposal', True)
            self.man.update_status(self.man.folderDict, 'selection_target_manual', True)
                
            if not self.man.folderDict['round_status']['selection_subset']:
                print('Doing selection_subset')
                # Query new samples
                self.query(opts, self.man.folderDict, self.man, NumSamples=opts.ALSamples[self.man.folderDict['version']-2])
                # Cre/mnt/hpc_XAL/data/UNetAL/CTA18/AL/tmp/USIMFTRCA_V09/maate action
                self.create_action(opts, self.man, self.man.datasets['action_round'].data, create_pseudo=True, create_mask=False,info_classified=True, info_selected=True, info_annotated=False)
                self.man.update_status(self.man.folderDict, 'selection_subset', True)
                sys.exit('Please annotate images using XALabeler software!')
                
            if not self.man.folderDict['round_status']['selection_annotation_manual']:
                print('Doing selection_annotation_manual')
                sys.exit('Please annotate images using XALabeler software!')
                self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)
                
            #annotated = self.annotation_manual(opts, man=self.man, data_action=self.man.datasets['action_round'].data)
            #if not annotated:
            #    sys.exit('Please annotate images using XALabeler software!')
            
            if not self.man.folderDict['round_status']['training']:
                print('Doing training')
                _ = self.annotation_manual(opts, man=self.man, data_action=self.man.datasets['action_round'].data)
                # Preprocessing raw data
                #subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
                subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -overwrite_plans_name " + opts.plans_identifier +  " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
                # Train model
                self.train_step(opts, self.man.folderDict)
                # Copy model and labeles
                self.copy_model_labels(opts, self.man.folderDict)
                self.man.update_status(self.man.folderDict, 'training', True)


        else:
            # Create new AL round  
            self.alcreate(opts)
            # Query new samples
            self.query(opts, self.man.folderDict, self.man, NumSamples=opts.ALSamples[self.man.folderDict['version']-2])
            # Annotate images
            self.annotation_auto(opts, self.man, self.man.datasets['action_round'].data)
            # Preprocessing raw data
            #subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -overwrite_plans_name " + opts.plans_identifier +  " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            # Train model
            self.train_step(opts, self.man.folderDict)
            # Copy model and labeles
            self.copy_model_labels(opts, self.man.folderDict)
            self.man.update_status(self.man.folderDict, 'training', True)
            # Update round_status
            for key in self.man.folderDict['round_status']:
                self.man.update_status(self.man.folderDict, key, True)

    def reset(self, opts, copy_nnUNet_data=True, key_reset='selection_annotation_manual'):
        
        # self=alunet
        # copy_nnUNet_data=True
        # key_reset='selection_annotation_manual'
        
        # Load current round
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        _ = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=copy_nnUNet_data)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round', 'action_pre'], load_class=opts.CLPatch, hdf5=False)
        opts.alunet=self
        
        keys = [k for k in self.man.folderDict['round_status']]
        keys.reverse()  
        for k in keys:  
            if k=='training':
                #self.man.folderDict['round_status'][k]=False
                self.man.update_status(self.man.folderDict, k, False)
            if k=='selection_annotation_manual':
                NumSamples=opts.ALSamples[self.man.folderDict['version']-2]
                self.man.datasets['action_round'].data = self.man.datasets['train'].data[-NumSamples:]
                self.man.datasets['train'].data = self.man.datasets['train'].data[0:-NumSamples]
                # Reset labelsTr
                labelsTr = join(nnUNet_raw, opts.DName, 'labelsTr')
                shutil.rmtree(labelsTr)
                labelsTr_pre = join(self.man.folderDict['modelpath_prev'], 'labelsTr')
                shutil.copytree(labelsTr_pre, labelsTr)
                # Create action for segmentation
                #self.create_action(opts, self.man, self.man.datasets['action_round'].data, classification=False, create_mask=False, create_pseudo=True)
                # Reset status
                #self.man.folderDict['round_status'][k]=False
                self.man.update_status(self.man.folderDict, k, False)
                # Save
                self.man.save(include=['train', 'action_round'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
            if k==key_reset:
                break
  
        
        
    def alround_target(self, opts, copy_nnUNet_data=True):
        
        # opts.label_manual = True
        # opts.strategy = 'RANDOM'
        # opts.ALSamples[folderDict['version']]=100
        # self=alunet
        # copy_nnUNet_data=False
        
        # keys -> corrected, preselected, classified, selected, annotated
        

        # Load current round
        NewVersion = False
        VersionUse = None
        strategy_name = opts.strategy
        self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        _ = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=copy_nnUNet_data)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round', 'action_pre'], load_class=opts.CLPatch, hdf5=False)
        
        if opts.fp_manual is None:
            fp_manual = join(os.path.dirname(self.man.folderDict['modelpath']), 'data_manual')
        else:
            fp_manual = opts.fp_manual
        NumSamples=opts.ALSamples[self.man.folderDict['version']-2]
        opts.alunet=self
        
        # Set parameter
        NumSamplesCheck = NumSamples
        NumSamplesPre = NumSamples*3
        
        # Create new training round
        processed = np.all([self.man.folderDict['round_status'][key] for key in self.man.folderDict['round_status']])
        if processed: 
            _, _ = self.alcreate(opts, copy_nnUNet_data=True)
            
            #self.man.datasets['query'].data=self.man.datasets['query'].data[0:500]

            # Precompute entropy and gradients
            if opts.fastmode:
                # Reset fastmode
                for s in self.man.datasets['query'].data + self.man.datasets['action_round'].data + self.man.datasets['action_pre'].data:
                    if 'grad' in s.F: del s.F['grad']
                    if 'FI' in s.F: del s.F['FI']
                    if 'uc' in s.F: del s.F['uc']
                st = USIMFT()
                st.getEntGrad(opts, self.man.folderDict, self.man, data_query=self.man.datasets['query'].data, batch_size=4, device='cuda', previous=True)
                self.man.save(include=['query'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
                #self.man.load(include=['query'], load_dict={}, load_class=opts.CLPatch, hdf5=False)
                
                
                #st = USIMFT()
                #st.getEntGrad(opts, self.man.folderDict, self.man, data_query=sl, batch_size=2, device='cuda', previous=True)
                
                #st.getEntGrad(opts, self.man.folderDict, self.man, data_query=self.man.datasets['train'].data, batch_size=2, device='cuda', previous=True)
                #self.man.save(include=['train'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
                
                #st.getEntGrad(opts, folderDict, self.man, data=self.man.datasets['train'].data, batch_size=4, device='cuda', previous=True)
                #self.man.save(include=['query', 'train'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
                # for s in self.man.datasets['query'].data:
                #     if 'grad' in s.F:
                #         del s.F['grad']
                #     if 'FI' in s.F:
                #         del s.F['FI']
        
        # Correction
        if opts.correction:
            # Correction proposal
            if not self.man.folderDict['round_status']['correction_proposal']: 
                print('Doing correction_proposal')
                #man, folderDict = self.alcreate(opts, copy_nnUNet_data=False)
                strategy = strategies[opts.strategy]()
                data_check = strategy.check_labeled(opts, self.man.folderDict, self.man, data_query=self.man.datasets['query'].data, data_train=self.man.datasets['train'].data, NumSamplesCheck=NumSamplesCheck, previous=True)
                self.man.datasets['action_round'].data = data_check
                self.man.save(include=['action_round'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
                self.create_action(opts, self.man, self.man.datasets['action_round'].data, classification=True, classification_multislice=False, create_mask=False, create_pseudo=True, pseude_is_mask=True, pseudo_full=False)
                self.man.update_status(self.man.folderDict, 'correction_proposal', True)
               
            # Correction selection (manual)
            if not self.man.folderDict['round_status']['correction_selection_manual']:
                print('Doing correction_selection_manual')
                sys.exit('Please select images for correction using XALabeler software!')
                self.man.update_status(self.man.folderDict, 'correction_selection_manual', True)
                
            # Correction preparation
            if not self.man.folderDict['round_status']['correction_preparation']:
                print('Doing correction_preparation')
                # Read action list
                fip_actionlist = os.path.join(fp_manual, 'actionlist.json')
                actionlist = ALAction.load(fip_actionlist) 
                #data_action = self.man.datasets['action_round'].data
                data_train = self.man.datasets['train'].data
                for a in actionlist:
                    for s in data_train:
                        if a.id==s.ID:
                            s.action = a
                            a.filetype='.nii.gz'
                data_action = [s for s in data_train if hasattr(s, "action")]
                # Select images for correction
                data_correct = [s for s in data_action if ((len(s.action.info['class'])>0) and (s.action.info['class'][-1][1]==1))]
                self.man.datasets['action_round'].data = data_correct
                self.man.save(include=['action_round'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
                self.create_action(opts, self.man, self.man.datasets['action_round'].data, classification=False, classification_multislice=False, create_mask=False, create_pseudo=True, pseude_is_mask=True, pseudo_full=True)
                self.man.update_status(self.man.folderDict, 'correction_preparation', True)
                sys.exit('Please correct images for correction using XALabeler software!')
                
            # Correction annotation (manual)
            if not self.man.folderDict['round_status']['correction_annotation_manual']:
                print('Doing correction_annotation_manual')
                #self.annotation_auto(opts, self.man, data=self.man.datasets['action_round'].data, full=False, update_train=False)
                _ = self.annotation_manual(opts, self.man, self.man.datasets['action_round'].data)
                # self.man.update_status(self.man.folderDict, 'correction_annotation_manual', True)
        else:
            self.man.update_status(self.man.folderDict, 'correction_proposal', True)
            self.man.update_status(self.man.folderDict, 'correction_selection_manual', True)
            self.man.update_status(self.man.folderDict, 'correction_preparation', True)
            self.man.update_status(self.man.folderDict, 'correction_annotation_manual', True)
            
                    
        # Selection proposal
        if not self.man.folderDict['round_status']['selection_proposal']:
            print('Doing selection_proposal')
            # Create new AL round  
            #self.alcreate(opts)
            fastmode = opts.fastmode
            opts.fastmode = False
            # Query new samples
            strat = opts.strategy
            opts.strategy='UCPROP'
            self.query(opts, self.man.folderDict, self.man, NumSamples=NumSamplesPre, NumSamplesMaxMCD=opts.NumSamplesMaxMCD)
            opts.strategy = strat
            self.man.datasets['action_pre'].data = self.man.datasets['action_round'].data
            self.man.datasets['action_round'].data=[]
            self.man.save(include=['action_round', 'action_pre'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
            # Create action
            self.create_action(opts, self.man, self.man.datasets['action_pre'].data, classification=True, classification_multislice=True, create_mask=False, create_pseudo=True, pseudo_full=True)
            self.man.update_status(self.man.folderDict, 'selection_proposal', True)
            opts.fastmode = fastmode

        # Selection of targets (manual)
        if not self.man.folderDict['round_status']['selection_target_manual']:
            print('Doing selection_target_manual')
            sys.exit('Please classifie images in positive and negative using XALabeler software!')
            self.man.update_status(self.man.folderDict, 'selection_target_manual', True)

        
        # Subset selection based on selected target samples
        if not self.man.folderDict['round_status']['selection_subset']:
            if opts.dim==2:
                print('Doing selection_subset')
                # Read action list
                fip_actionlist = os.path.join(fp_manual, 'actionlist.json')
                actionlist = ALAction.load(fip_actionlist) 
                action_pre = self.man.datasets['action_pre'].data
                for a in actionlist:
                    for s in action_pre:
                        if a.id==s.ID:
                            s.action = a
                            a.filetype='.nii.gz'
                action_pre = [s for s in action_pre if hasattr(s, "action")]
    
                # Check if samples are classified as positive and negative
                classes = [(s.F['imagename'], s.action.info['class']) for s in action_pre if 'class' in s.action.info]
                df_classes = pd.DataFrame()
                for cl in classes:
                    imname = cl[0]
                    for sl in cl[1]:
                        #df_classes=df_classes.append({'imagename': imname, 'slice': sl[0], 'class': sl[1]}, ignore_index=True)
                        df_classes = pd.concat([df_classes, pd.DataFrame([{'imagename': imname, 'slice': sl[0], 'class': sl[1]}])], ignore_index=True)
    
                
                data_query = self.man.datasets['query'].data
                data_action_select = []
                for index, row in df_classes.iterrows():
                    for s in action_pre+data_query:
                        if (s.F['imagename']==row['imagename']) and (s.F['lbs_org'][0]==row['slice']):
                            s.action = ALAction()
                            s.action.info['class']=row['class']
                            data_action_select.append(s)
            else:
                data = self.man.datasets['action_round'].data + self.man.datasets['action_pre'].data + self.man.datasets['query'].data
                file1 = open(os.path.join(fp_manual, 'XALabelerLUT.ctbl'), 'r')
                lines = file1.readlines()
                for line in lines:
                    if line.split(' ')[1]=='positive':
                        label_positive=int(line.split(' ')[0])
                    if line.split(' ')[1]=='negative':
                        label_negative=int(line.split(' ')[0])
                        
                imagenames = sorted(np.unique([s.F['imagename'] for s in data]))
                data_action_select=[]
                for imagename in imagenames:
                    fip_label = join(fp_manual, 'refine', imagename + '.nii.gz')
                    if os.path.isfile(fip_label):
                        #sys.exit()
                        ref = CTRef(fip_label).ref()
                        label_pos = skimage.measure.label(ref==label_positive)
                        props_pos = skimage.measure.regionprops_table(label_pos, properties=['centroid', 'area', 'label'])
                        idx_pos = np.ones((props_pos['centroid-0'].shape[0]))*-1
                        dist_pos = np.ones((props_pos['centroid-0'].shape[0]))*1000000
                        label_neg = skimage.measure.label(ref==label_negative)
                        props_neg = skimage.measure.regionprops_table(label_neg, properties=['centroid', 'area', 'label'])                      
                        idx_neg = np.ones((props_neg['centroid-0'].shape[0]))*-1
                        dist_neg = np.ones((props_neg['centroid-0'].shape[0]))*1000000
                        for i,s in enumerate(data):
                            if s.F['imagename']==imagename:
                                #sys.exit()
                                bL=s.F['lbs_org']
                                bU=s.F['ubs_org']
                                x = np.mean([bL[0],bU[0]])
                                y = np.mean([bL[1],bU[1]])
                                z = np.mean([bL[2],bU[2]])
                                # Positive samples
                                distp = ((x-props_pos['centroid-0'])*(x-props_pos['centroid-0']))+((y-props_pos['centroid-1'])*(y-props_pos['centroid-1']))+((z-props_pos['centroid-2'])*(z-props_pos['centroid-2']))
                                idxp = np.where(distp<dist_pos)
                                dist_pos[idxp] = distp[idxp]
                                idx_pos[idxp] = i
                                # Negative samples
                                distn = ((x-props_neg['centroid-0'])*(x-props_neg['centroid-0']))+((y-props_neg['centroid-1'])*(y-props_neg['centroid-1']))+((z-props_neg['centroid-2'])*(z-props_neg['centroid-2']))
                                idxn = np.where(distn<dist_neg)
                                dist_neg[idxn] = distn[idxn]
                                idx_neg[idxn] = i
                        
                        idx_pos = np.unique(idx_pos)
                        idx_neg = np.unique(idx_neg)
                        for i in list(idx_pos):
                            s = data[int(i)]
                            s.action = ALAction()
                            s.action.info['class']=1
                            data_action_select.append(s)
                        for i in list(idx_neg):
                            s = data[int(i)]
                            s.action = ALAction()
                            s.action.info['class']=2
                            data_action_select.append(s)


                # data = self.man.datasets['action_pre'].data + self.man.datasets['query'].data
                # #label_ignore = load_json(os.path.join(nnUNet_preprocessed, opts.DName, 'dataset.json'))['labels']['ignore']
                # file1 = open(os.path.join(fp_manual, 'XALabelerLUT.ctbl'), 'r')
                # lines = file1.readlines()
                # for line in lines:
                #     if line.split(' ')[1]=='positive':
                #         label_positive=int(line.split(' ')[0])
                #     if line.split(' ')[1]=='negative':
                #         label_negative=int(line.split(' ')[0])
                        
                
                # dataSort = UNetPatchV2.sort_F(data, prop='imagename')
                # data_action_select=[]
                # for i,s in enumerate(dataSort):
                #     if i==0 or (dataSort[i-1].F['imagename']!=dataSort[i].F['imagename']):
                #         fip_label = join(fp_manual, 'refine', dataSort[i].F['imagename'] + '.nii.gz')
                #         if os.path.isfile(fip_label):
                #             ref = CTRef(fip_label).ref()
                #         else:
                #             ref = None
                #     if ref is not None:
                #         #sys.exit()
                #         bL=dataSort[i].F['lbs_org']
                #         bU=dataSort[i].F['ubs_org']
                #         arr = ref[bL[0]:bU[0], bL[1]:bU[1], bL[2]:bU[2]]
                #         #if (arr==label_ignore).sum()>0:
                #         #    data_action_select.append(s)
                #         if (arr==label_positive).sum()>0:
                #             s.action = ALAction()
                #             s.action.info['class']=1
                #             data_action_select.append(s)
                #         if (arr==label_negative).sum()>0:
                #             s.action = ALAction()
                #             s.action.info['class']=2
                #             data_action_select.append(s)
                            
                # Select random subset
                # idx = [x for x in range(len(data_action_select))]
                # random.shuffle(idx)
                # idx = idx[0:NumSamples]
                # data_action_select = [data_action_select[i] for i in idx]
                    
                 
            #self.man.datasets['query'].delete(data_action)
            #data_query = self.man.datasets['query'].data
            
            # !!!
            #del self.man.datasets['query'].data[0].F['grad']

            
            data_select = self.select_manual(opts, self.man.folderDict, self.man, data_action_select, NumSamples=NumSamples)
            self.man.datasets['query'].delete(data_select)
            #self.man.datasets['action_pre'].data = self.man.datasets['action_round'].data
            self.man.datasets['action_round'].data = data_select
            self.man.save(include=['action_round', 'train','query'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
            # Create action
            self.create_action(opts, self.man, self.man.datasets['action_round'].data, classification=False, create_mask=False, create_pseudo=True)
            self.man.update_status(self.man.folderDict, 'selection_subset', True)
            
            # for s in data_pos:
            #     print(s.F['imagename'])
                      
        
        #print('opts123', opts)
        # Annotated images (manual)
        if not self.man.folderDict['round_status']['selection_annotation_manual']:
            print('Doing selection_annotation_manual')
            if opts.segauto:
                self.annotation_auto(opts, self.man, data=self.man.datasets['action_round'].data, full=False)
                self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)
            else:
                sys.exit('Please annotate images using XALabeler software!')
                #_ = self.annotation_manual(opts, self.man, self.man.datasets['action_round'].data, bg_offset=False)
                #self.annotation_auto(opts, self.man, data=self.man.datasets['action_round'].data, full=False)
                #sys.exit('Please annotate images using XALabeler software!')
                self.man.update_status(self.man.folderDict, 'selection_annotation_manual', True)
                
        # Reset fastmode
        if opts.fastmode:
            for s in self.man.datasets['query'].data + self.man.datasets['action_round'].data + self.man.datasets['action_pre'].data:
                if 'grad' in s.F: del s.F['grad']
                if 'FI' in s.F: del s.F['FI']
                if 'uc' in s.F: del s.F['uc']
            self.man.save(include=['query', 'action_round', 'action_pre'], save_class=opts.CLPatch, hdf5=False)
            
        if not self.man.folderDict['round_status']['training']:
            print('Doing training')
            
            #print('train123', len(self.man.datasets['train'].data))
            #print('action_round123', len(self.man.datasets['action_round'].data))
            #sys.exit()
            
            #self.man.datasets['action_round'].data = self.man.datasets['train'].data[-50:]
            #self.man.datasets['train'].data = self.man.datasets['train'].data[0:-50]

            # !!! Replace ND (19) with background (0) in refined labels
            # fp_manual = join(os.path.dirname(self.man.folderDict['modelpath']), 'data_manual')
            # fp_refine = join(fp_manual, 'refine')
            # ignore = load_json(os.path.join(nnUNet_raw, opts.DName, 'dataset.json'))['labels']['ignore']
            # ND = load_json(os.path.join(nnUNet_raw, opts.DName, 'dataset.json'))['labels']['ND']
            # fip_refines = glob(fp_refine + '/*')
            # for fip_refine in fip_refines:
            #     ref = CTRef(fip_refine)
            #     arr = ref.ref()
            #     arr[arr==ND]=ignore
            #     ref.setRef(arr)
            #     ref.save(fip_refine)            
            
            # Annotate images
            _ = self.annotation_manual(opts, self.man, self.man.datasets['action_round'].data, bg_offset=False)
            

            # sys.exit('EXIT_TEST')
            # Preprocessing raw data
            #subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            # !!!
            subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -overwrite_plans_name " + opts.plans_identifier +  " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
            # Train model
            self.train_step(opts, self.man.folderDict, pretrained=True)
            # Copy model and labeles
            self.copy_model_labels(opts, self.man.folderDict)
            self.man.update_status(self.man.folderDict, 'training', True)
            
        
            
    
    def alfull(self, opts):
        
        # self=alunet
        # Create new AL round  
        NewVersion = True
        VersionUse = None
        strategy_name = opts.strategy
        man = self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        folderDict = man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True)
        man.load(include=['train', 'query', 'valid', 'action', 'action_round'], load_class=opts.CLPatch, hdf5=False)
        
        # Update training set
        man.datasets['train'].data = man.datasets['query'].data
        man.save(include=['train, query'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
        
        # Update annotation
        if opts.label_manual:
            pass
            #self.annotation_manual(man.datasets['train'].data)
        else:
            self.annotation_auto(opts, man, data=man.datasets['train'].data, full=True)

        # Preprocessing raw data
        #subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
        subprocess.call("nnUNetv2_plan_and_preprocess -d " + opts.dataset_name_or_id + " -overwrite_plans_name " + opts.plans_identifier +  " --verify_dataset_integrity -c " + opts.configuration + " --clean", shell=True)
        
        # Train model
        self.train(opts)
        
        # Copy model and labeles
        self.copy_model_labels(opts, folderDict)
        
    def alexp(self, opts, preprocess=False):
        
        # self = alunet
        NumSamples = 50
        
        # Load current round
        NewVersion = False
        VersionUse = None
        copy_nnUNet_data = False
        strategy_name = opts.strategy
        self.man = opts.CLManager(fp_dataset=opts.dataset_data)
        _ = self.man.createALFolderpath(opts, method=strategy_name, NewVersion=NewVersion, VersionUse=VersionUse, copy_prev_labelsTr=True, copy_nnUNet_data=copy_nnUNet_data)
        self.man.load(include=['train', 'query', 'valid', 'action', 'action_round', 'action_pre'], load_class=opts.CLPatch, hdf5=False)
        
        if opts.fp_manual is None:
            fp_manual = join(os.path.dirname(self.man.folderDict['modelpath']), 'data_manual')
        else:
            fp_manual = opts.fp_manual
            
        data = self.man.datasets['action_round'].data + self.man.datasets['action_pre'].data + self.man.datasets['query'].data
        file1 = open(os.path.join(fp_manual, 'XALabelerLUT.ctbl'), 'r')
        lines = file1.readlines()
        for line in lines:
            if line.split(' ')[1]=='positive':
                label_positive=int(line.split(' ')[0])
            if line.split(' ')[1]=='negative':
                label_negative=int(line.split(' ')[0])
                
        imagenames = sorted(np.unique([s.F['imagename'] for s in data]))
        data_action_select=[]
        for imagename in imagenames:
            fip_label = join(fp_manual, 'refine', imagename + '.nii.gz')
            if os.path.isfile(fip_label):
                print('imagename', imagename)
                #sys.exit()
                ref = CTRef(fip_label).ref()
                label_pos = skimage.measure.label(ref==label_positive)
                props_pos = skimage.measure.regionprops_table(label_pos, properties=['centroid', 'area', 'label'])
                idx_pos = np.ones((props_pos['centroid-0'].shape[0]))*-1
                dist_pos = np.ones((props_pos['centroid-0'].shape[0]))*1000000
                label_neg = skimage.measure.label(ref==label_negative)
                props_neg = skimage.measure.regionprops_table(label_neg, properties=['centroid', 'area', 'label'])                      
                idx_neg = np.ones((props_neg['centroid-0'].shape[0]))*-1
                dist_neg = np.ones((props_neg['centroid-0'].shape[0]))*1000000
                for i,s in enumerate(data):
                    if s.F['imagename']==imagename:
                        #sys.exit()
                        bL=s.F['lbs_org']
                        bU=s.F['ubs_org']
                        x = np.mean([bL[0],bU[0]])
                        y = np.mean([bL[1],bU[1]])
                        z = np.mean([bL[2],bU[2]])
                        # Positive samples
                        distp = ((x-props_pos['centroid-0'])*(x-props_pos['centroid-0']))+((y-props_pos['centroid-1'])*(y-props_pos['centroid-1']))+((z-props_pos['centroid-2'])*(z-props_pos['centroid-2']))
                        idxp = np.where(distp<dist_pos)
                        dist_pos[idxp] = distp[idxp]
                        idx_pos[idxp] = i
                        # Negative samples
                        distn = ((x-props_neg['centroid-0'])*(x-props_neg['centroid-0']))+((y-props_neg['centroid-1'])*(y-props_neg['centroid-1']))+((z-props_neg['centroid-2'])*(z-props_neg['centroid-2']))
                        idxn = np.where(distn<dist_neg)
                        dist_neg[idxn] = distn[idxn]
                        idx_neg[idxn] = i
                
                idx_pos = np.unique(idx_pos)
                idx_neg = np.unique(idx_neg)
                for i in list(idx_pos):
                    s = data[int(i)]
                    s.action = ALAction()
                    s.action.info['class']=1
                    data_action_select.append(s)
                for i in list(idx_neg):
                    s = data[int(i)]
                    s.action = ALAction()
                    s.action.info['class']=2
                    data_action_select.append(s)
                print('data_action_select', len(data_action_select))
                #sys.exit()

    
        #data_select = self.select_manual2(opts, self.man.folderDict, self.man, data_action_select, NumSamples=NumSamples)
        
        previous=True 
        strategy = strategies[opts.strategy]()
        self.man.alunet = self
        data_select = strategy.select_manual2(opts, self.man.folderDict, self.man, self.man.datasets['query'].data, data_action_select, NumSamples)
        
        
        self.man.datasets['query'].delete(data_select)
        #self.man.datasets['action_pre'].data = self.man.datasets['action_round'].data
        self.man.datasets['action_round'].data = data_select
        self.man.save(include=['action_round', 'train','query'], save_dict={}, save_class=opts.CLPatch, hdf5=False)
        # Create action
        self.create_action(opts, self.man, self.man.datasets['action_round'].data, classification=False, create_mask=False, create_pseudo=True)
        self.man.update_status(self.man.folderDict, 'selection_subset', True)

            
        
    def info(self, opts):
        # Plot information with respect to performance
        pass 
    
    
    @staticmethod    
    def load_val(method='macro', datasets=['Spleen', 'BrainTumour'], strategies=[], fp_UNetAL='/mnt/HHD/data/', fn='UNetAL', diceAll=False):
        
        if method=='macro':
            read_logfile=False
            micro = False
        elif method=='logfile':
            read_logfile=True
            micro = False
        elif method=='micro':
            read_logfile=False
            micro = True
            
        
        fp_UNetALF = fp_UNetAL+fn
        if len(strategies)==0: 
            strategies = ['RANDOM', 'USIMF', 'USIMFR']

        dval=dict()
        for dim in ['2D', '3D']:
            if dim=='2D':
                d_dataset={}
                for dataset in datasets:
                    fp_dataset = os.path.join(fp_UNetALF, dataset, 'AL')
                    d_strategy={}
                    for strategy in strategies:
                        fp_strategy = os.path.join(fp_dataset, strategy)
                        if os.path.isdir(fp_strategy):
                            fp_versions = sorted(glob(fp_strategy+'/*'))
                            d_version={}
                            for fp_version in fp_versions:
                                if len(glob(os.path.join(fp_version, 'model')+'/nnUNetTrainer*'))>0:
                                    fp_plan = glob(os.path.join(fp_version, 'model')+'/nnUNetTrainer*')[0]
                                    fp_folds = sorted(glob(fp_plan+'/fold*'))
                                    d_fold={}
                                    for fp_fold in fp_folds:
                                        if read_logfile:
                                            fip_logs = glob(fp_fold+'/training*')
                                            fip_logs.sort(key=os.path.getmtime)
                                            fip_log = fip_logs[-1]
                                            #sys.exit()
                                            if os.path.isfile(fip_log):
                                                with open(fip_log) as f:
                                                    lines = f.readlines()
                                                line_val_all = [l for l in lines if 'Yayy' in l]
                                                if len(line_val_all)>0:
                                                    line_val = line_val_all[-1]
                                                    dice = float(line_val.split('Dice: ')[-1].split(' \n')[0])
                                                    d_fold[os.path.basename(fp_fold)]=dice
                                                    d_version[os.path.basename(fp_version)]=d_fold
                                        else:
                                            fp_summary = os.path.join(fp_fold,'validation', 'summary.json')
                                            #sys.exit()
                                            if os.path.isfile(fp_summary):
                                                with open(fp_summary, 'r') as d:
                                                    summary = json.load(d)
                                                #dice = float(summary['mean']['1']['Dice'])
                                                if not micro:
                                                    if diceAll:
                                                        dmean = summary['mean']
                                                        dice = [dmean[x]['Dice'] for x in dmean]
                                                        dice = dice +[float(summary['foreground_mean']['Dice'])]
                                                    else:
                                                        dice = float(summary['foreground_mean']['Dice'])
                                                    d_fold[os.path.basename(fp_fold)]=dice
                                                    d_version[os.path.basename(fp_version)]=d_fold
                                                else:
                                                    TP=0
                                                    FP=0
                                                    FN=0
                                                    NumVal = len(summary['metric_per_case'])
                                                    for ca in summary['metric_per_case']:
                                                        for cl in ca['metrics'].keys():
                                                            TP=TP+ca['metrics'][cl]['TP']
                                                            FP=FP+ca['metrics'][cl]['FP']
                                                            FN=FN+ca['metrics'][cl]['FN']
                                                    dice = (2*TP)/(2*TP+FP+FN)
                                                    d_fold[os.path.basename(fp_fold)]=dice
                                                    d_version[os.path.basename(fp_version)]=d_fold
                                            
                            d_strategy[strategy]=d_version
                    d_dataset[dataset]=d_strategy
                dval[dim]=d_dataset
        return dval      

    @staticmethod    
    def load_test(opts, fp_UNetAL, datasets=['Spleen', 'BrainTumour'], strategies=[], keys=['images-clDice_micro-dRCA','images-clDice_micro-RPDA', 'images-clDice_micro-RPLB']):
        # keys=['images-clDice_micro-dRCA','images-clDice_micro-RPDA', 'images-clDice_micro-RPLB']
        # keys=['images-clDice_micro-dLAD']
        # keys=['images-clDice']
        # keys=['images-PREC-dLAD']
        # keys=['images-RECALL-dLAD']
        # keys=['images-PREC-dRCA', 'images-PREC-RPDA', 'images-PREC-RPLB']
        # keys=['images-RECALL-dRCA', 'images-RECALL-RPDA', 'images-RECALL-RPLB']
        # keys=['images-RECALL-dRCA']
        # keys=['images-F1_micro']
        # keys=['images-clPrec_micro-RPLB']
        # keys=['images-clSens_micro-RPLB']
        # keys=['images-clPrec_micro-RPLB', 'images-clSens_micro-RPLB']
        # keys=['images-clPrec_micro-RPDA', 'images-clSens_micro-RPDA']
        # keys=['images-clPrec_micro-pLAD', 'images-clSens_micro-pLAD']
        # keys=['images-clDice_micro']
        # keys=['images-clDice_micro-dLAD', 'images-PREC-dLAD', 'images-RECALL-dLAD']
        # keys=['images-clDice_micro-dRCA', 'images-PREC-dRCA', 'images-RECALL-dRCA']
        # keys=['images-clDice_micro-pRCA', 'images-PREC-pRCA', 'images-RECALL-pRCA']
        
        # keys=['images-clDice_micro-dLAD', 'images-clPrec_micro-dLAD', 'images-clSens_micro-dLAD']
        
        from os.path import join
        import pandas as pd
        d_dataset={}
        for dataset in datasets:
            fp_dataset = os.path.join(fp_UNetAL, dataset, 'AL')
            d_strategy={}
            for strategy in strategies:
                fp_strategy = os.path.join(fp_dataset, strategy)
                fp_init = os.path.join(fp_dataset, 'USIMFT')
                if os.path.isdir(fp_strategy):
                    fp_versions = sorted(glob(fp_init+'/*'))[0:4] + sorted(glob(fp_strategy+'/*'))[1:]
                    d_version={}
                    for key in keys:
                        d_version[key]={}
                        for fp_version in fp_versions:
                            fip_perform = join(fp_version, 'test', 'perform.pkl')
                            if os.path.isfile(fip_perform):
                                df_perform = pd.read_pickle(fip_perform)
                                if len(key.split('-'))==1:
                                    value = df_perform[key]
                                    d_version[key][os.path.basename(fp_version)]=value
                                elif len(key.split('-'))==2:
                                    values = [im[key.split('-')[1]] for im in df_perform[key.split('-')[0]]]
                                    values = [v for v in values if v is not None]
                                    value = np.mean(values)
                                    d_version[key][os.path.basename(fp_version)]=value
                                elif len(key.split('-'))==3:
                                    values = [im[key.split('-')[1]][key.split('-')[2]] for im in df_perform[key.split('-')[0]]]
                                    values = [v for v in values if v is not None]
                                    value = np.mean(values)
                                    d_version[key][os.path.basename(fp_version)]=value

  
                
class UNetManagerV2(ALManager):
    """
    UNetManagerV2
    """
    def __init__(self, fp_dataset):
        ALManager.__init__(self, fp_dataset)
        self.fp_dataset = fp_dataset
        for key in self.datasets.keys():
            self.datasets[key] = UNetDatasetV2(key)
        self.datasets['action']=UNetDatasetV2('action')
        self.datasets['action_round']=UNetDatasetV2('action_round')
        self.datasets['action_pre']=UNetDatasetV2('action_pre')
        
    def createALFolderpath(self, opts, fip_split=None, method=None, NewVersion=False, VersionUse=None, copy_prev_labelsTr=False, copy_nnUNet_data=True):

        folderDict = super().createALFolderpath(fp_active=opts.fp_active, fip_split=fip_split, fp_images=opts.fp_images, fp_references_org=opts.fp_references_org, method=method, NewVersion=NewVersion, VersionUse=VersionUse)
        #folderDict['fip_hdf5_all'] = os.path.join(opts.fp_active, 'hdf5_all_'+str(opts.dim)+'D.hdf5')
        # Copy nnunet dataset
        if folderDict['copy_init']:
            dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
            fp_nnunet = opts.fp_nnunet
            fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
            fp_nnUNet_preprocessed = os.path.join(fp_nnunet, 'nnUNet_preprocessed')
            fp_nnUNet_results = os.path.join(fp_nnunet, 'nnUNet_results')
            #dnameInit = 'Dataset' + '1'+opts.dataset_name_or_id[1:2]+'0' + '_' + opts.dataset
            dnameInit = 'Dataset' + opts.dataset_name_or_id[0]+'00' + '_' + opts.dataset
            if copy_nnUNet_data:
                print('Copy nnUNet_raw')
                copy_tree(os.path.join(fp_nnUNet_raw, dnameInit), os.path.join(fp_nnUNet_raw, dname))
                print('Copy nnUNet_preprocessed')
                copy_tree(os.path.join(fp_nnUNet_preprocessed, dnameInit), os.path.join(fp_nnUNet_preprocessed, dname))
                print('Copy nnUNet_result')
                copy_tree(os.path.join(fp_nnUNet_results, dnameInit), os.path.join(fp_nnUNet_results, dname))
            
            # Change dataset_name in nnUUnetPlans.json
            plans_file = os.path.join(nnUNet_preprocessed, dname, opts.plans_identifier+'.json')
            plans = load_json(plans_file)
            plans['dataset_name'] = dname
            save_json(plans, plans_file)
            
        # Reset labelsTr from prvious round
        
        if folderDict['version']>1 and folderDict['newVersion'] and copy_prev_labelsTr:
            dname = 'Dataset' + opts.dataset_name_or_id + '_' + opts.dataset
            fp_nnunet = opts.fp_nnunet
            fp_nnUNet_raw = os.path.join(fp_nnunet, 'nnUNet_raw')
            fp_nnunetData = os.path.join(fp_nnUNet_raw, dname)
            fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')
            fp_labelsTr_prev = os.path.join(folderDict['modelpath_prev'], 'labelsTr')
            if os.path.exists(fp_labelsTr):
                shutil.rmtree(fp_labelsTr)
            shutil.copytree(fp_labelsTr_prev, fp_labelsTr)
            
        # Store round infomation
        folderDict['round_status']=OrderedDict({'correction_proposal': False,
                                                'correction_selection_manual': False, 
                                                'correction_preparation': False, 
                                                'correction_annotation_manual': False,
                                                'selection_proposal': False, 
                                                'selection_target_manual': False, 
                                                'selection_subset': False, 
                                                'selection_annotation_manual': False,
                                                'training': False})
        folderDict['fip_round_status'] = join(folderDict['fp_manager'], 'round_status.json')
        if os.path.isfile(folderDict['fip_round_status']) and not NewVersion:
            with open(folderDict['fip_round_status'], 'r') as f:
                folderDict['round_status'] = json.load(f)
        else:
            with open(folderDict['fip_round_status'], 'w') as json_file:
                json.dump(folderDict['round_status'], json_file, indent = 4)
            

        # Update path for nnUNet
        #ALFolderpathMethod = os.path.join(fp_active, method)
        #if not os.path.isdir(ALFolderpathMethod) and not method=='INIT':
        #folderDict['modelpath'] = 
        self.folderDict=folderDict
        return folderDict
    
    def update_status(self, folderDict, key, value):
        if key not in folderDict['round_status']:
            raise ValueError("Key " + key + " not found in status dict.")
        folderDict['round_status'][key]=value
        with open(folderDict['fip_round_status'], 'w') as json_file:
            json.dump(folderDict['round_status'], json_file, indent = 4)
    
    def init_patches(self, opts):
        
        # self=self.man
        
        tile_step_size=0.5
        #images_train = load_json(join(nnUNet_preprocessed, opts.DName, 'splits_final.json'))[opts.fold]['train']
        images = glob(join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'_'+opts.configuration + '/*.npz'))
        patch_size = load_json(join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'.json'))['configurations'][opts.configuration]['patch_size']
        if len(patch_size)==2: patch_size=[1]+patch_size
        patch_size = np.array(patch_size).astype(np.int32)
        plans_file = os.path.join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'.json')
        with open(plans_file, 'r') as f:
            plans = json.load(f)
        spacing_after_resampling = plans['configurations'][opts.configuration]['spacing']
        if len(spacing_after_resampling)==2: spacing_after_resampling=[1.0]+spacing_after_resampling
        dim = opts.dim
        ID=0
        data=[]
        for image in tqdm(images):
            #imname = os.path.basename(image).split('.')[0]
            imname = splitFilePath(image)[1]
            #if 'BUD-008' in imname:
            #     sys.exit()
            #sys.exit()
            pkl = pd.read_pickle(join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'_'+opts.configuration, imname+'.pkl'))
            fip = os.path.join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'_' + opts.configuration, image)
            #shape_resample = np.load(fip)['seg'].shape[4-opts.dim:]
            #shape_resample = np.load(fip)['seg'].shape[4-opts.dim:]
            shape_after_resampling = np.load(fip)['seg'].shape[1:]
            #steps = compute_steps_for_sliding_window(pkl['shape_before_cropping'], patch_size, tile_step_size)
            steps = compute_steps_for_sliding_window(shape_after_resampling, patch_size, tile_step_size)
            # Steps have to be adjusted from 1 to 2 because the sliding window with 0.5 overlab would create multiple images from same slice
            if opts.dim==2:
                steps[0] = [i for i in range(shape_after_resampling[0])]
            
            #if len(steps)==2: steps=steps+[[None]]
            for sx in list(steps[0]):
                for sy in list(steps[1]):
                    for sz in list(steps[2]):
                        #sys.exit()
                        
                        if opts.dim==2:
                            need_to_pad = [patch_size[i+1] - shape_after_resampling[i+1] for i in range(dim)]
                            lbs = [- need_to_pad[i] // 2 for i in range(dim)]
                            #ubs = [shape_after_resampling[i+1] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - patch_size[i+1] for i in range(dim)]
                            coord = np.array([[0,int(sx+patch_size[0]/2),int(sy+patch_size[1]/2),int(sz+patch_size[2]/2)]])
                            selected_voxel = coord[0][2:]
                            bbox_lbs = [max(lbs[i], selected_voxel[i] - patch_size[i+1] // 2) for i in range(dim)]
                            bbox_ubs = [bbox_lbs[i] + patch_size[i+1] for i in range(dim)]
                            #valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
                            #valid_bbox_ubs = [min(shape_after_resampling[i+1], bbox_ubs[i]) for i in range(dim)]
                            padding = [(0,0)]+[(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape_after_resampling[i+1], 0)) for i in range(dim)]
                          
                            patch = UNetPatchV2()
    
                            patch.ID=ID
                            patch.F['props']=dict()
                            patch.F['ID']=ID
                            #patch.F['imagename']=os.path.basename(image).split('.')[0]
                            patch.F['imagename']=splitFilePath(image)[1]
                            patch.F['classLocations']=coord
                            patch.F['spacing']=np.array(pkl['spacing'])
                            patch.F['shape_before_cropping']=pkl['shape_before_cropping']
                            patch.F['bbox_used_for_cropping']=pkl['bbox_used_for_cropping']
                            patch.F['shape_after_cropping_and_before_resampling']=pkl['shape_after_cropping_and_before_resampling']
                            patch.F['shape_after_resampling']=shape_after_resampling
                            patch.F['patch_size']=patch_size
                            patch.F['spacing_after_resampling']=spacing_after_resampling
                            patch.F['padding']=padding
                            # Defined properties
                            ratio=patch.F['spacing_after_resampling']/patch.F['spacing']
                            if opts.dim==2:
                                ratio[0]=1.0
                            patch.F['lbs_res'] = np.array([sx+padding[0][0], sy+padding[1][0], sz+padding[2][0]]).astype(np.int32)
                            patch.F['ubs_res'] = np.array([sx+patch_size[0]-padding[0][1], sy+patch_size[1]-padding[1][1], sz+patch_size[2]-padding[2][1]]).astype(np.int32)
                            patch.F['lbs_crop'] = (np.round(patch.F['lbs_res']*ratio)).astype(np.int32)
                            patch.F['ubs_crop'] = (np.round(patch.F['ubs_res']*ratio)).astype(np.int32)
                            patch.F['lbs_org'] = np.array([patch.F['lbs_crop'][i]+pkl['bbox_used_for_cropping'][i][0] for i in range(dim+1)])
                            patch.F['ubs_org'] = np.array([patch.F['ubs_crop'][i]+pkl['bbox_used_for_cropping'][i][0] for i in range(dim+1)])
    
                            # Update minimum and maximum lbs_crop and ubs_crop
                            patch.F['lbs_org'] = np.maximum(patch.F['lbs_org'], np.zeros(patch.F['lbs_org'].shape, int))
                            patch.F['ubs_crop'] = np.minimum(patch.F['ubs_org'], np.array(patch.F['shape_before_cropping']))
                        
                        else:
                            need_to_pad = [patch_size[i] - shape_after_resampling[i] for i in range(dim)]
                            lbs = [- need_to_pad[i] // 2 for i in range(dim)]
                            #ubs = [shape_after_resampling[i+1] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - patch_size[i+1] for i in range(dim)]
                            coord = np.array([[0,int(sx+patch_size[0]/2),int(sy+patch_size[1]/2),int(sz+patch_size[2]/2)]])
                            selected_voxel = coord[0][1:]
                            bbox_lbs = [max(lbs[i], selected_voxel[i] - patch_size[i] // 2) for i in range(dim)]
                            bbox_ubs = [bbox_lbs[i] + patch_size[i] for i in range(dim)]
                            #valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
                            #valid_bbox_ubs = [min(shape_after_resampling[i+1], bbox_ubs[i]) for i in range(dim)]
                            padding = [(0,0)]+[(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape_after_resampling[i], 0)) for i in range(dim)]
                
                            patch = UNetPatchV2()
                            patch.ID=ID
                            patch.F['props']=dict()
                            patch.F['ID']=ID
                            #patch.F['imagename']=os.path.basename(image).split('.')[0]
                            patch.F['imagename']=splitFilePath(image)[1]
                            patch.F['classLocations']=coord
                            patch.F['spacing']=np.array(pkl['spacing'])
                            patch.F['shape_before_cropping']=pkl['shape_before_cropping']
                            patch.F['bbox_used_for_cropping']=pkl['bbox_used_for_cropping']
                            patch.F['shape_after_cropping_and_before_resampling']=pkl['shape_after_cropping_and_before_resampling']
                            patch.F['shape_after_resampling']=shape_after_resampling
                            patch.F['patch_size']=patch_size
                            patch.F['spacing_after_resampling']=spacing_after_resampling
                            patch.F['padding']=padding
                            # Defined properties
                            ratio=patch.F['spacing_after_resampling']/patch.F['spacing']
                            if opts.dim==2:
                                ratio[0]=1.0
                            patch.F['lbs_res'] = np.array([sx+padding[0][0], sy+padding[1][0], sz+padding[2][0]]).astype(np.int32)
                            patch.F['ubs_res'] = np.array([sx+patch_size[0]-padding[0][1], sy+patch_size[1]-padding[1][1], sz+patch_size[2]-padding[2][1]]).astype(np.int32)
                            patch.F['lbs_crop'] = (np.round(patch.F['lbs_res']*ratio)).astype(np.int32)
                            patch.F['ubs_crop'] = (np.round(patch.F['ubs_res']*ratio)).astype(np.int32)
                            patch.F['lbs_org'] = np.array([patch.F['lbs_crop'][i]+pkl['bbox_used_for_cropping'][i][0] for i in range(dim)])
                            patch.F['ubs_org'] = np.array([patch.F['ubs_crop'][i]+pkl['bbox_used_for_cropping'][i][0] for i in range(dim)])
    
                            # Update minimum and maximum lbs_crop and ubs_crop
                            patch.F['lbs_org'] = np.maximum(patch.F['lbs_org'], np.zeros(patch.F['lbs_org'].shape, int))
                            patch.F['ubs_crop'] = np.minimum(patch.F['ubs_org'], np.array(patch.F['shape_before_cropping']))
                        
                        data.append(patch)   
                        ID=ID+1

        # Perform 5-fold split
        from nnunetv2.run.run_training import get_trainer_from_args
        nnunet_trainer = get_trainer_from_args(opts.dataset_name_or_id, opts.configuration, opts.fold, opts.nnUNetTrainer, opts.plans_identifier, False)
        split = nnunet_trainer.do_split()
        
        # Copy split file
        fip_split_nnunet = os.path.join(nnUNet_preprocessed, opts.DName, 'splits_final.json')
        shutil.copyfile(fip_split_nnunet, opts.fip_split)
        
        # Split validation data
        data_split = json.load(open(opts.fip_split))
        images_valid = data_split[opts.fold]['val']
        data_valid=[]
        data_query=[]
        for s in data:
            #if s.F['imagename'].split('.')[0] in images_valid:
            if s.F['imagename'] in images_valid:
                data_valid.append(s)
            else:
                data_query.append(s)
        self.datasets['valid'].data = data_valid
        self.datasets['query'].data = data_query
        
        
    def load_model(self, opts, folderDict, previous=False, const_dropout=False, load_weights=True):
        # self=man

        dataset_name_or_id = opts.dataset_name_or_id
        configuration = opts.configuration
        tr=opts.nnUNetTrainer
        p=opts.plans_identifier
        use_compressed='False'
        from modules.nnUNet.code.nnunetv2.run.run_training import get_trainer_from_args
        nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, opts.fold, tr, p, use_compressed)
        settingsfilepath_model = os.path.join(folderDict['modelpath'], 'LITS.yml')
        net = ALUNETMODEL2(settingsfilepath_model, overwrite=True)
        net.model = {'unet': nnunet_trainer}
        if previous:
            net.props['fip_checkpoint']=folderDict['modelpath_prev']
        else:
            net.props['fip_checkpoint']=folderDict['modelpath']
        if load_weights:
            #net.model['unet'].load_checkpoint(os.path.join(net.props['fip_checkpoint'], opts.nnUNetResults, 'fold_'+str(opts.fold), 'checkpoint_best.pth'))
            net.model['unet'].load_checkpoint(os.path.join(net.props['fip_checkpoint'], opts.nnUNetResults, 'fold_'+str(opts.fold), 'checkpoint_final.pth'))
            print('Loading model:', os.path.join(net.props['fip_checkpoint']))
        else:
            net.model['unet'].initialize()
        return net      
    
    

    
    
class UNetDatasetV2(SALDataset):
    """
    UNetDatasetV2
    """
    def __init__(self, name=''):
        SALDataset.__init__(self, name)
            
class UNetSampleV2(ALSample):
    """
    UNetSampleV2
    """
    def __init__(self):
        ALSample.__init__(self)
        self.Xlabel = ['XImage']
        self.Ylabel = ['XMask']
        self.Plabel = ['XMaskPred']
        self.patches = []
               
        
    @classmethod
    def sort_F(cls, sl, prop='imagename', prop_sorted=False):
        if prop_sorted:
            props = sorted(np.unique([s.F[prop] for s in sl]))
        else:
            props = [s.F[prop] for s in sl]
            indexes = np.unique(props, return_index=True)[1]
            props = [props[index] for index in sorted(indexes)]
        slout=[]
        for pr in props:
            for s in sl:
                if s.F['imagename']==pr:
                    slout.append(s)
        return slout

    @classmethod
    def savePseudo(cls, opts, data, folderDict, fp_pseudo, filetype, pseudo_full=False, tile_step_size=0.5):
        # cls=ALSegmentCACSSample
        # data=data_samples
        
        dataSort = UNetPatchV2.sort_F(data, prop='imagename')
        
        plans = load_json(join(nnUNet_preprocessed, opts.DName, opts.plans_identifier+'.json'))
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(opts.configuration)
        
        #idx = np.argsort([s.F['imagename'] for s in dataSort])
        fp_nnunetData = os.path.join(nnUNet_raw, opts.DName)
        fp_imagesTr = os.path.join(fp_nnunetData, 'imagesTr')
        fp_labelsTr = os.path.join(fp_nnunetData, 'labelsTr')
        #imagenames = list(np.sort([s.F['imagename'] for s in dataSort]))
        imagenames = list([s.F['imagename'] for s in dataSort])
        
        if pseudo_full:
            imagenames = list(np.unique(imagenames))
            for i,imn in enumerate(tqdm(imagenames, desc='Save pseudo label')):
                print('fp_imagesTr123', fp_imagesTr)
                print('imn123', imn)
                fip_image = glob(fp_imagesTr + '/'+imn+'_*')[0]
                print('fip_image123', fip_image)
                fip_pseudo = os.path.join(fp_pseudo, imn + filetype)
                modelpath = opts.alunet.man.folderDict['modelpath_prev']
                opts.alunet.predict_image(opts, folderDict, modelpath, fip_image, fip_pseudo, tile_step_size=tile_step_size)
        
        else:
            for i,imn in enumerate(tqdm(imagenames, desc='Save pseudo label')):
                s = dataSort[i]
                if i==0 or imagenames[i]!=imagenames[i-1]:
                    #fip_pseudo = glob(join(fp_labelsTr, imn)+'.*')[0]
                    fip_pseudo = join(fp_labelsTr, imn+filetype)
                    ref = CTRef(fip_pseudo)
                    arr = ref.ref()
                    arr = arr*0
    
                original_spacing = s.F['spacing_after_resampling']
                target_spacing = s.F['spacing']
                padding = s.F['padding']
                # Reshape pseudo label
                XMaskPred = s.P['XMaskPred']
                # Invert padding
                XMaskPredPad = XMaskPred[:,padding[0][0]:XMaskPred.shape[1]-padding[0][1],padding[1][0]:XMaskPred.shape[2]-padding[1][1], padding[2][0]:XMaskPred.shape[3]-padding[2][1]]
                # Invert rescale
                # !!!!
                if s.F['ubs_org'][1]>512:
                    s.F['ubs_org'][1]=512
                if s.F['ubs_org'][2]>512:
                    s.F['ubs_org'][2]=512
                shape_tar_seg = (XMaskPredPad.shape[1], s.F['ubs_org'][1]-s.F['lbs_org'][1], s.F['ubs_org'][2]-s.F['lbs_org'][2])
                XMaskPredProp = torch.from_numpy(configuration_manager.resampling_fn_data(XMaskPredPad, shape_tar_seg, original_spacing, target_spacing))
                XMaskPredHot = torch.argmax(compute_one_hot_torch(XMaskPredProp), dim=1)
                
                arr[s.F['lbs_org'][0]:s.F['ubs_org'][0], s.F['lbs_org'][1]:s.F['ubs_org'][1], s.F['lbs_org'][2]:s.F['ubs_org'][2]]=XMaskPredHot
                
                if i==len(imagenames)-1 or imn!=imagenames[i+1]:
                    ref_out = CTRef(arr)
                    ref_out.copyInformationFrom(ref)
                    name_pseudo = imn + filetype
                    fip_pseudo = os.path.join(fp_pseudo, name_pseudo)
                    ref_out.save(fip_pseudo)

        
class UNetPatchV2(UNetSampleV2):
    """
    UNetPatchV2
    """

    def __init__(self):
        UNetSampleV2.__init__(self)
        #self.IDP=None
        self.F['props']=None
        
    @classmethod
    def save(cls, sl, fp_data, save_dict, dataset_name='', hdf5=False):
        if len(sl)>0:
            if hdf5:
                cls.save_hdf5(sl=sl, fp_data=fp_data, save_dict=save_dict, dataset_name=dataset_name)
            else:
                cls.save_pkl(sl=sl, fp_data=fp_data, save_dict=save_dict, dataset_name=dataset_name)

    @classmethod
    def save_pkl(cls, sl, fp_data, save_dict, dataset_name='', hdf5=False):
        os.makedirs(fp_data, exist_ok=True)
        df = pd.DataFrame()
        fip_df = os.path.join(fp_data, 'sl.pkl')
        features = [s.F.copy() for s in tqdm(sl)]
        df = pd.DataFrame.from_records(features)
        df.to_pickle(fip_df)
        

    @classmethod
    def load(cls, fp_data, load_dict, load_class, dataset_name='', hdf5=False):
        if hdf5:
            return cls.load_hdf5(fp_data=fp_data, load_dict=load_dict, load_class=load_class, dataset_name=dataset_name)
        else:
            return cls.load_pkl(fp_data=fp_data, load_dict=load_dict, dataset_name=dataset_name)

    @classmethod
    def load_pkl(cls, fp_data, load_dict, dataset_name=''):
        data=[]
        fip_df = os.path.join(fp_data, 'sl.pkl')
        if os.path.isfile(fip_df):
            df = pd.read_pickle(fip_df)
            for index, row in df.iterrows():
                s = cls()
                s.ID = row['ID']
                s.name = str(row['ID'])
                s.F=dict(row)
                data.append(s)
        return data

    def __plot_image(self, image, name, title, color=False, save=False, filepath=None, format_im=None, dpi=300):
        if color:
            #print('image1234', image.shape)
            im = np.zeros((image.shape[2], image.shape[3]))
            for c in range(image.shape[1]):
                im = im + (c+1) * image[0,c,:,:]
            plt.imshow(im)
            plt.imshow(im, cmap='Accent', interpolation='nearest')
            if title: plt.title(name) 
            if save: plt.savefig(filepath + name + format_im, format=format_im, dpi=dpi)
            plt.show()
        else:
            if title: plt.title(name) 
            if save: plt.savefig(filepath + name + format_im, format=format_im, dpi=dpi)
            plt.imshow(image[0,0,:,:], cmap='gray')
            plt.show()
            
    def plotSample(self, plotlist=['XImage', 'XMask', 'XPred', 'XWeight', 'XRegion', 'XPseudo', 'XRefine'], save=False, fp='', name='', color=False, title=True, format_im='svg', dpi=300):
        self.cpu()
        filepath = os.path.join(fp, name)

        for d in self.da:
            if d in plotlist:
                plotlist = plotlist + list(getattr(self, d, None).keys())

        for pl in plotlist:
            name = self.name + '_' + pl  + '_' + str(int(self.ID))
            im = self.getXY(pl)
            if im is not None:
                image = im.data.numpy()
                print('image.shape123', image.shape)
                if pl=='XImage' and len(image.shape)==4:
                    idx = int((image.shape[1]-1)/2)
                    image = image[:,idx:idx+1]
                    self.__plot_image(image, name, title, color=False, filepath=filepath, save=save)
                else:
                    #image = image[:,1:2]
                    self.__plot_image(image, name, title, color=color, filepath=filepath, save=save)
    
    def showImageJ(self, plot='XImage'):
        if plot=='XImage':
            arr=self.X['XImage'][0,0,:,:,:].detach().cpu().numpy()
        elif plot=='XMaskPred':
            arr=np.zeros((self.P['XMaskPred'].shape[2], self.P['XMaskPred'].shape[3], self.P['XMaskPred'].shape[4]))
            for i in range(self.P['XMaskPred'].shape[1]):
                arr=arr+self.P['XMaskPred'][0,i,:,:,:].detach().cpu().numpy()*i
        else:
            print('Plot method ' + plot + ' not defined!')
            return
        im = CTImage(arr)
        im.showImageJ()
        
                    
class ALUNETMODEL2(DLBaseModel):
    
    """
    ALUNETMODEL2 model
    """

    def __init__(self, settingsfilepath, overwrite=False):
        props = defaultdict(lambda: None,
            NumChannelsIn = 1,
            NumChannelsOut = 2,
            Input_size = (512, 512, 1),
            Output_size = (512, 512, 2),
            device = 'cuda',
            modelname = 'UNetSeg',
            savePretrainedEpochMin=0
        )
        DLBaseModel.__init__(self, settingsfilepath=settingsfilepath, overwrite=overwrite, props=props)

