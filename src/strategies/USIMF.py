import os, sys
from os.path import join
# Set system path
sys.path.append('/mnt/SSD2/cloud_data/Projects/CTP/src/')
sys.path.append('/mnt/SSD2/cloud_data/Projects/CTP/src/tools/nngeometry')
sys.path.append('/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/tools/nngeometry')
import random
import torch
from tqdm import tqdm
import math
import numpy as np
from nngeometry.layercollection import LayerCollection
import pandas as pd
from glob import glob
from modules.XAL.strategies.ALStrategy import ALStrategy
from modules.UNetAL.ALUNet import UNetPatch
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from submodlib.functions.facilityLocationVariantMutualInformation import FacilityLocationVariantMutualInformationFunction
from submodlib.functions.facilityLocationConditionalMutualInformation import FacilityLocationConditionalMutualInformationFunction
from submodlib.functions.logDeterminantConditionalMutualInformation import LogDeterminantConditionalMutualInformationFunction
from batchgenerators.utilities.file_and_folder_operations import load_json
from kneed import KneeLocator
from ct.ct import CTImage
from sklearn.neighbors import LocalOutlierFactor
soft1 = torch.nn.Softmax(dim=1)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def enable_dropout(model, drop_rate=0.5):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.p=drop_rate
            m.train()
        if m.__class__.__name__.startswith('ConvDropoutNormReLU'):
            m.p=drop_rate
            m.train()
            
def disable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.p=0.0
            m.eval()
        if m.__class__.__name__.startswith('ConvDropoutNormReLU'):
            m.p=0.0
            m.eval()

def similarity_fim(data0, data1, FI):
    S=torch.zeros((len(data0), len(data1)))
    G0 = torch.vstack([s.F['grad']*FI.cpu() for s in data0])
    G1 = torch.vstack([s.F['grad'] for s in data1]).transpose(0,1)
    S = torch.matmul(G0,G1)/len(FI)
    Sn = S.numpy()
    return Sn
             
            
class USIMF(ALStrategy):
    def __init__(self, name='USIMF'):
        self.name = name
        self.dtype=torch.FloatTensor

    def query(self, opts, folderDict, man, data_query, NumMCD=10, NumSamples=10, pred_class='XRegionPred', batchsize=100, previous=True, save_uc=False, NumSamplesMaxMCD=50000):
        
        # self=strategy

        # Define parameters
        layer_used='decoder'
        NumSamplesQ=NumSamples
        NumParams=10000
        NumFIMMax=200                    # 500
        save_uc=False
        dropout_rate=0.1
        NumMCDRounds=10
        batch_size=4

        # Compute uncertainty
        data_query = random.sample(data_query, k=min(len(data_query), NumSamplesMaxMCD))
        self.getUncertainty(opts, folderDict, man, data_query, batch_size=batch_size, save_uc=save_uc, device='cuda', previous=True, NumMCDRounds=NumMCDRounds, dropout_rate=dropout_rate)
        
        # Extract weights
        classWeight = self.getClassWeighting(opts, man)

        # Weighted uncertainty
        label_ignore = load_json(join(nnUNet_preprocessed, opts.DName, 'dataset.json'))['labels']['ignore']
        uc = np.array([s.F['uc'] for s in data_query])
        ucw = np.zeros(uc.shape[0])
        for i in range(label_ignore):
            ucw = ucw+classWeight[i]*uc[:,i]
        prop = ucw/ucw.sum()
        
        # Get Network and layers
        net, layer_collection, layers = self.getNetLayer(opts, man, folderDict, layer_used, previous=True)

        # Compute F
        data_F = list(np.random.choice(data_query, size=min(NumFIMMax, len(data_query)), replace=False, p=prop))
        FI, idxG = self.computeF(opts, man, net, layer_collection, folderDict, data_F, layer_used, NumParams, previous=True)

        # Compute gradients of data_QU and data_uc
        self.grad_data(opts, folderDict, man, net, data_query, layer_collection, previous, idxG=idxG)

        # Estimate number of cluster
        x = np.array([i for i in range(len(uc))])
        y = sorted(ucw)[::-1]
        kneedle = KneeLocator(x, y, S=10.0, curve="convex", direction="decreasing")
        NumSamplesQ = int(min(max(kneedle.knee, NumSamplesQ), len(data_query)))

        # Submodular subset selection
        data_Q = list(np.random.choice(data_query, size=min(NumSamplesQ, len(data_query)), replace=False, p=prop))
        query_sijs = similarity_fim(data_query, data_Q, FI)
        n = len(data_query)
        num_queries = len(data_Q)
        optimizer = 'StochasticGreedy'
        stopIfZeroGain = False
        stopIfNegativeGain = False
        verbose = False
        show_progress = False
        epsilon = 0.1
        budget = NumSamples
        queryDiversityEta = 0.01
        obj = FacilityLocationVariantMutualInformationFunction(n, num_queries, query_sijs=query_sijs, data=None, queryData=None, metric=None, queryDiversityEta=queryDiversityEta)
        greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, stopIfNegativeGain=stopIfNegativeGain, verbose=verbose, show_progress=show_progress, epsilon=epsilon)
        greedyIndices = [x[0] for x in greedyList]
        samples = [data_query[i] for i in greedyIndices]
        
        for s in samples:
            print(s.F['imagename'], s.F['lbs_org'][0])
        
        # Delete gradients and uncertainty
        for s in samples + data_query + data_Q + data_F:
            if 'grad' in s.F: del s.F['grad']
            if 'uc' in s.F: del s.F['uc']
  
        return samples
    
    def getClassWeighting(self, opts, man):
        data_train = man.datasets['train'].data
        imagenames = np.unique([s.F['imagename'] for s in data_train])
        label_ignore = load_json(os.path.join(nnUNet_preprocessed, opts.DName, 'dataset.json'))['labels']['ignore']
        weight = np.zeros(label_ignore)
        for im in imagenames:
            fip_image = glob(os.path.join(nnUNet_raw, opts.DName, 'labelsTr', im)+'*')[0]
            arr = CTImage(fip_image).image()
            for c in range(label_ignore):   
                weight[c] = weight[c] + (arr==c).sum()/arr.size
        weight = 1/weight
        weight[0] = 0
        classWeight = weight/weight.sum()
        return classWeight
        
        
    def grad_data(self, opts, folderDict, man, net, data, layer_collection, previous, idxG=None, idx_class=None, append=False, fisherG=True, use_mask=False):
        
        # data = data_query
        # idxG=[1,2,3,4,5]
        
        def loss_fn(out, target):
            pred_weak_bin_log = torch.log_softmax(out, dim=1)
            pred_weak_bin_prop = torch.exp(pred_weak_bin_log)
            loss=[]
            for c in range(pred_weak_bin_prop.shape[1]):
                loss.append(torch.mean(pred_weak_bin_log[:,c] * target[:,c]))
            loss = torch.mean(torch.stack(loss))
            return loss

        def compute_grad(sample, sample_target):
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
            grad_params = torch.autograd.grad(loss, list(net.model['unet'].network.parameters()), allow_unused=True)
            gradv = torch.hstack([torch.reshape(grad, (-1,)) for grad in grad_params if grad is not None])
            return gradv
        
        def compute_sample_grads(data, target):
            """ manually process each sample with per sample gradient """
            sample_grads = [compute_grad(data[i], target[i]) for i in range(data.shape[0])]
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
 
            datab = datab.to(device, non_blocking=True)
            per_sample_grads = compute_sample_grads(datab, target)
            
            # Filter gradients
            if idxG is not None:
                per_sample_grads=[v[idxG] for v in per_sample_grads ]
            
            # Set gradient in data
            for i,ID in enumerate(list(IDX)):
                data[ID].F['grad']=per_sample_grads[i].detach().cpu().double()



    def getNetLayer(self, opts, man, folderDict, layer_used, previous):
        # Extract layers
        net = man.load_model(opts, folderDict, previous=previous)
        if layer_used=='mid':
            for layer, mod in net.model['unet'].network.named_modules():
                if 'encoder' in layer and 'convs' in layer and '1.conv' in layer:
                    mid_layer=layer
            layers=[mid_layer]
        elif layer_used=='decoder':
            layers=[]
            for layer, mod in net.model['unet'].network.named_modules():
                if 'decoder' in layer and 'convs' in layer and ('0.conv' in layer or '1.conv' in layer):
                    layers.append(layer)
        elif layer_used=='encoder_decoder':
            layers=[]
            for layer, mod in net.model['unet'].network.named_modules():
                if 'encoder' in layer and 'convs' in layer and ('0.conv' in layer or '1.conv' in layer):
                    layers.append(layer)
            for layer, mod in net.model['unet'].network.named_modules():
                if 'decoder' in layer and 'convs' in layer and ('0.conv' in layer or '1.conv' in layer):
                    layers.append(layer)
        elif layer_used=='decoder_mid':
            layers=[]
            for layer, mod in net.model['unet'].network.named_modules():
                if 'decoder' in layer and 'convs' in layer and ('0.conv' in layer or '1.conv' in layer):
                    layers.append(layer)
            for layer, mod in net.model['unet'].network.named_modules():
                if 'encoder' in layer and 'convs' in layer and '1.conv' in layer:
                    mid_layer=layer
            layers.append(mid_layer)
        elif layer_used=='last':
            for layer, mod in net.model['unet'].network.named_modules():
                if 'decoder' in layer and 'convs' in layer and '1.conv' in layer:
                    lastlayer = layer
            layers=[lastlayer]
        elif layer_used=='manual':
            layers=[]
            layers.append('decoder.stages.4.convs.1.conv')
            layers.append('decoder.stages.5.convs.1.conv') 
            layers.append('decoder.stages.6.convs.1.conv') 
        layer_collection = LayerCollection.from_model_BF(net.model['unet'].network, ignore_unsupported_layers=True,layer_in=layers)
        return net, layer_collection, layers


    def computeF(self, opts, man, net, layer_collection, folderDict, data_F, layer_used, NumParams, previous, idxG=None, idx_class=None, fisherG=False):
        
        eps=1e-20

        # Compute gradients
        self.grad_data(opts, folderDict, man, net, data_F, layer_collection, previous, idxG=idxG, idx_class=idx_class, fisherG=fisherG)
        params = layer_collection.get_parameters_BF(net.model['unet'].network)
        num_params = int(np.sum([x.numel() for x in params]))
        gradEmb = torch.zeros(data_F[0].F['grad'].shape)
        for s in data_F:
            gradEmb = gradEmb+(s.F['grad']*s.F['grad'])
        gradEmb = (gradEmb/len(data_F)).numpy().astype('float64')
        population = [x for x in range(len(gradEmb))]
        prop = gradEmb/gradEmb.sum()
        NumParamsSel = min(NumParams, (gradEmb>eps).sum())
        if idxG is None:
            idxG = np.random.choice(population, size=min(num_params, NumParamsSel), replace=False, p=prop)
            F=torch.from_numpy(gradEmb[idxG])
        else:
            F=torch.from_numpy(gradEmb)
        FI=(1/F).cuda()
        return FI, idxG


    def getUncertainty(self, opts, folderDict, man, data_query, batch_size=None, save_uc=False, device='cuda', previous=True, NumMCDRounds=10, dropout_rate=0.01):

        # self=strategy
        
        # init model
        net = man.load_model(opts, folderDict, previous=previous)
        net.model['unet'].network.eval()
        enable_dropout(net.model['unet'].network, dropout_rate)
        
        dataloader_train = net.model['unet'].get_dataloaders_alunet(data_load=data_query, batch_size=batch_size)
        NumBatches = math.ceil(len(data_query)/dataloader_train.data_loader.batch_size)
        uc=[]
        for b in tqdm(range(NumBatches), desc='Estimate uncertainty'):
            batch = next(dataloader_train)
            IDX = batch['idx'][:,0]
            datab = batch['data']
            datab = datab.to(device, non_blocking=True)
            
            # Iterate over monte carlo rounds
            MCDrounds=[]
            for r in range(NumMCDRounds):
                out = net.model['unet'].network(datab)
                for i in range(len(out)): out[i] = out[i].detach_().cpu()
                outs = soft1(out[0])
                MCDrounds.append(torch.unsqueeze(outs, dim=0).clone())
            MCDrounds = torch.vstack(MCDrounds)
            ucMap = torch.var(MCDrounds, dim=0)
            dimMean = tuple([i for i in range(2,2+opts.dim)])
            uc = torch.mean(ucMap, dim=dimMean).numpy()
            
            # Set uncertainties in data
            for i,ID in enumerate(list(IDX)):
                data_query[ID].F['uc']=uc[i]
                if save_uc: 
                    data_query[ID].F['ucMap']=ucMap[i]

