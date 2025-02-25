#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
sys.path.insert(0,"/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/modules/nnUNet/code")
sys.path.append("/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src")
sys.path.append("/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/modules/nnUNet/code/nnunetv2")
import argparse
import socket
from utils.ALUNETV2 import ALUNETV2
hostname = socket.gethostname()

# python TAL.py --dataset=CTA18 --dataset_name_or_id=900 --strategy=USIMFT --func=al --dim=3 --fold=1 --configuration=3d_fullres --budget=-1 --label_manual=True --targeted=True --correction=False --segauto=False --fastmode=True --versionUse=-1
# python TAL.py --dataset=CTA18 --dataset_name_or_id=900 --strategy=USIMFT --func=al --label_manual=True --targeted=True --segauto=False  --versionUse=-1

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Dataset name', type=str, default='CTA18')
    parser.add_argument('--dataset_name_or_id', help='Dataset name', type=str, default='900')
    parser.add_argument('--strategy', help='AL strategy', type=str, default='USIMFTRCA')
    parser.add_argument('--emulation', help='Function to execute', type=str, default=False)
    parser.add_argument('--func', help='Function to execute', type=str, default='')
    parser.add_argument('--dim', help='Number of dimensions (2D or 3D model)', type=int, default=3)
    parser.add_argument('--fold', help='Cross validation fold number', type=int, default=0)
    parser.add_argument('--configuration', help='Configuration', type=str, default='3d_fullres')
    parser.add_argument('--fp_raw', help='Folderpath to raw data', type=str, default='/mnt/HHD/data/UNetAL/data_raw/Hippocampus')
    parser.add_argument('--nnUNetTrainer', help='Name of the nnUNetTrainer', type=str, default='nnUNetTrainer_ALUNET')
    parser.add_argument('--targeted', help='Use targeted selection selection', type=str2bool, default=False)
    parser.add_argument('--label_manual', help='Label manual', type=str2bool, default=True)
    parser.add_argument('--correction', help='Label correction', type=str2bool, default=False)
    parser.add_argument('--segauto', help='Automatic segmentation', type=str2bool, default=True)
    parser.add_argument('--serverdata', help='Automatic segmentation', type=str2bool, default=False)
    parser.add_argument('--fastmode', help='Fast computation', type=str2bool, default=True)
    parser.add_argument('--label_valid', help='Label validation set', type=str2bool, default=False)
    parser.add_argument('--versionUse', help='AL version', type=int, default=None)

    opts = parser.parse_args()
    opts_dict = vars(opts)
    strategy_dict = {'INIT': 0,
                     'RANDOM': 2,
                     'USIMFT': 4,
                     'USIMF': 6,
                     'MCD': 11,
                     'USIMFTRCA': 19,
                     'USIMFTLAD': 22,
                     'USIMFTSIDE': 24,
                     'BADGE': 27}

    opts.dataset_name_or_id=str(int(opts.dataset_name_or_id)+strategy_dict[opts.strategy])
    VersionUse = opts.versionUse
    
    if hostname=='ANONYMIZED' and not opts.serverdata:
        opts_dict['dataset_data'] = os.path.join('/mnt/SSD2/cloud_data/Projects/CTP/src/modules', opts.dataset, 'data')
        opts_dict['fp_active'] = os.path.join('/mnt/HHD/data/UNetAL', opts.dataset, 'AL')
        opts_dict['fp_modules'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules'
        opts_dict['fip_split'] = os.path.join(opts_dict['fp_active'], 'INIT', 'INIT_V01', 'model', 'splits_final.json')
        opts_dict['fp_nnunet'] = '/mnt/HHD/data/UNetAL/nnunet'
        opts_dict['fp_raw'] = '/mnt/HHD/data/UNetAL/data_raw'  
        #opts_dict['fp_manual'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XALabeler/XALabeler/data_manual'  
        opts_dict['fp_manual'] = None
        opts_dict['nnUNetResults'] = opts_dict['nnUNetTrainer']+'__ALUNETPlanner__'+opts_dict['configuration']
        opts_dict['fp_nngeometry'] = '/mnt/SSD2/cloud_data/Projects/CTP/src/tools/nngeometry'
        opts_dict['ALSamples'] = [100 for i in range(10)]
        opts_dict['nnUNet_raw'] = '/mnt/HHD/data/UNetAL/nnunet/nnUNet_raw'
        opts_dict['nnUNet_preprocessed'] = '/mnt/HHD/data/UNetAL/nnunet/nnUNet_preprocessed'
        opts_dict['nnUNet_results'] = '/mnt/HHD/data/UNetAL/nnunet/nnUNet_results'
    else:
        opts_dict['dataset_data'] = os.path.join('/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/modules', opts.dataset, 'data')
        opts_dict['fp_active'] = os.path.join('/sc-projects/sc-proj-cc06-ag-dewey/data/UNetAL/', opts.dataset, 'AL')
        opts_dict['fp_modules'] = '/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/modules'
        opts_dict['fip_split'] = os.path.join(opts_dict['fp_active'], 'INIT', 'INIT_V01', 'model', 'splits_final.json')
        opts_dict['fp_nnunet'] = '/sc-projects/sc-proj-cc06-ag-dewey/data/UNetAL/nnunet'
        opts_dict['fp_raw'] = '/sc-projects/sc-proj-cc06-ag-dewey/data/UNetAL/data_raw'  
        opts_dict['fp_manual'] = None
        opts_dict['nnUNetResults'] = opts_dict['nnUNetTrainer']+'__ALUNETPlanner__'+opts_dict['configuration']
        opts_dict['fp_nngeometry'] = '/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/tools/nngeometry'
        opts_dict['ALSamples'] = [100 for i in range(10)]
        opts_dict['nnUNet_raw'] = '/sc-projects/sc-proj-cc06-ag-dewey/data/UNetAL/nnunet/nnUNet_raw'
        opts_dict['nnUNet_preprocessed'] = '/sc-projects/sc-proj-cc06-ag-dewey/data/UNetAL/nnunet/nnUNet_preprocessed'
        opts_dict['nnUNet_results'] = '/sc-projects/sc-proj-cc06-ag-dewey/data/UNetAL/nnunet/nnUNet_results'
    
    if opts.dataset=='CTA18':
        opts_dict['ALSamples'] = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
        opts.NumSamplesMaxMCD = 5000
    if opts.dataset=='LIVG2':
        opts_dict['ALSamples'] = [25, 25, 25, 25, 25, 25, 25, 25, 35, 25, 25, 25]
        opts.NumSamplesMaxMCD = 5000

        
    # Init ALUNETV2
    alunet = ALUNETV2(opts)

    # Init dataset
    if opts.func=='init': 
        opts.strategy='INIT'
        opts.dataset_name_or_id=str(int(opts.dataset_name_or_id))
        alunet.init_dataset(opts)
        
    # Active learning round
    if opts.func=='al' and opts.strategy!='FULL': 
        for i in range(1):
            alunet.alround(opts, copy_nnUNet_data=True)


