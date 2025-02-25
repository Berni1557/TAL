import os, sys
from os.path import join
import numpy as np
import torch
from torch import nn
from scipy import stats
from tqdm import tqdm
import random
from nngeometry.layercollection import LayerCollection
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from sklearn.cluster import KMeans
import pdb
from scipy import stats
import pandas as pd
import math
import time
import torch.nn.functional as F
from helper.DataAccess import DataAccess
from modules.XAL.strategies.ALStrategy import ALStrategy
from batchgenerators.utilities.file_and_folder_operations import load_json
sys.path.append('/mnt/SSD2/cloud_data/Projects/CTP/src/tools/nngeometry')
sys.path.append('/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/tools/nngeometry')


class BADGE(ALStrategy):
    def __init__(self, name='BADGE'):
        self.name = name


    def init_centers(self, X, K, device='cpu'):
        pdist = nn.PairwiseDistance(p=2)
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        #print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
                D2 = torch.flatten(D2)
                D2 = D2.cpu().numpy().astype(float)
            else:
                newD = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
                newD = torch.flatten(newD)
                newD = newD.cpu().numpy().astype(float)
                for i in range(len(X)):
                    if D2[i] >  newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
    
            #if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2)/ sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1

        return indsAll
    
    def query(self, opts, folderDict, man, data_query, CLSample, NumSamples=10, pred_class='XMaskPred', batchsize=None, save_uc=False, device='cuda', previous=True, NumSamplesMaxMCD=50000):
        
        # self=strategy
       
        net = man.load_model(opts, folderDict, previous=previous)
        net.model['unet'].network.eval()
        
        #layers=['decoder.seg_layers.4']
        layers=[]
        for name, module in net.model['unet'].network.named_modules():
            if 'decoder.seg_layers' in name:
                layers.append(name)
        layer_collection = LayerCollection.from_model_BF(net.model['unet'].network, ignore_unsupported_layers=True,layer_in=layers)
        data_query = random.sample(data_query, k=min(len(data_query), NumSamplesMaxMCD))
        self.grad_data(opts, folderDict, man, net, data_query, layer_collection, previous, idxG=None)
        gradEmb = torch.zeros(len(data_query), data_query[0].F['grad'].shape[0])

        # Fill array
        for i in range(len(data_query)):
            gradEmb[i,:]=data_query[i].F['grad']

        # Compute centers
        idx = self.init_centers(gradEmb.numpy(), NumSamples)
        
        # Create samples
        samples=[data_query[i] for i in idx]

        
        return samples

    def grad_data(self, opts, folderDict, man, net, data, layer_collection, previous, idxG=None, idx_class=None, append=False, fisherG=True, use_mask=False):
        
        # data = data_query

        def loss_fn(out, target):
            pred_weak_bin_log = torch.log_softmax(out, dim=1)
            pred_weak_bin_prop = torch.exp(pred_weak_bin_log)
            loss=[]
            for c in range(pred_weak_bin_prop.shape[1]):
                loss.append(torch.mean(pred_weak_bin_log[:,c] * target[:,c]))
            loss = torch.mean(torch.stack(loss))
            return loss

        def compute_grad(sample, sample_target, layer_collection):
            sample = sample.unsqueeze(0)  # prepend batch dimension for processing
            pred = net.model['unet'].network(sample)[0]
            pred_weak_bin_log = torch.log_softmax(pred, dim=1)
            pred_weak_bin_prop = torch.exp(pred_weak_bin_log)
            n_output = pred_weak_bin_prop.shape[1]
            if sample_target is None:
                #print('pred_weak_bin_prop123', pred_weak_bin_prop.shape)
                if opts.dim==2:
                    sample_target = torch.nn.functional.one_hot(torch.argmax(pred_weak_bin_prop, dim=1, keepdims=False),n_output).permute(0, 3, 1, 2)
                else:
                    sample_target = torch.nn.functional.one_hot(torch.argmax(pred_weak_bin_prop, dim=1, keepdims=False),n_output).permute(0, 4, 1, 2, 3)
            else:
                if opts.dim==2:
                    sample_target = torch.nn.functional.one_hot(sample_target,n_output).permute(0, 3, 1, 2)
                else:
                    sample_target = torch.nn.functional.one_hot(sample_target,n_output+1).permute(0, 4, 1, 2, 3)
                    sample_target = torch.reshape(sample_target, (1, n_output+1, -1))
                    pred = torch.reshape(pred, (1, n_output, -1))
                    idx = torch.where(sample_target[0,n_output,:]==0)[0]
                    sample_target = sample_target[:,0:n_output,idx]
                    pred = pred[:,:,idx]
            loss = loss_fn(pred, sample_target)
            
            params = layer_collection.get_parameters_BF(net.model['unet'].network)
            grad_params = torch.autograd.grad(loss, list(params), allow_unused=True)
            gradv = torch.hstack([torch.reshape(grad, (-1,)) for grad in grad_params if grad is not None])
            return gradv
        
        def compute_sample_grads(data, target, layer_collection):
            """ manually process each sample with per sample gradient """
            sample_grads = [compute_grad(data[i], target[i], layer_collection) for i in range(data.shape[0])]
            return sample_grads

        batch_size=32
        net = man.load_model(opts, folderDict, previous=True)
        net.model['unet'].network.eval()
        dataloader_train = net.model['unet'].get_dataloaders_alunet(data_load=data, batch_size=batch_size)
        NumBatches = math.ceil(len(data)/dataloader_train.data_loader.batch_size)
        device='cuda'
        label_ignore = load_json(join(nnUNet_preprocessed, opts.DName, 'dataset.json'))['labels']['ignore']
        for b in tqdm(range(NumBatches), desc='Compte gradient'):
            batch = next(dataloader_train)
            
            datab = batch['data']
            if use_mask:
                target=batch['target'][0].long().to(device)
                target[target==label_ignore]=0

            else:
                target=[None for i in range(datab.shape[0])]
            IDX = batch['idx'][:,0]
            #sys.exit()
            

            datab = datab.to(device, non_blocking=True)
            per_sample_grads = compute_sample_grads(datab, target, layer_collection)
            
            # Filter gradients
            if idxG is not None:
                per_sample_grads=[v[idxG] for v in per_sample_grads ]
            
            # Set gradient in data
            for i,ID in enumerate(list(IDX)):
                data[ID].F['grad']=per_sample_grads[i].detach().cpu().double()
