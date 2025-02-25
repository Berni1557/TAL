
import os, sys
# Set system path
sys.path.append('/mnt/SSD2/cloud_data/Projects/CTP/src/')
sys.path.append('/mnt/SSD2/cloud_data/Projects/CTP/src/tools/nngeometry')
sys.path.append('/sc-projects/sc-proj-cc06-ag-dewey/code/CTP/src/tools/nngeometry')
import random
import torch
from tqdm import tqdm
import math
import numpy as np
from glob import glob
from modules.XAL.strategies.ALStrategy import ALStrategy
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from batchgenerators.utilities.file_and_folder_operations import load_json
from ct.ct import CTImage
soft1 = torch.nn.Softmax(dim=1)

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
             
            
class MCD(ALStrategy):
    def __init__(self, name='MCD'):
        self.name = name
        self.dtype=torch.FloatTensor

    def query(self, opts, folderDict, man, data_query, NumMCD=10, NumSamples=10, pred_class='XRegionPred', batchsize=100, previous=True, save_uc=False, NumSamplesMaxMCD=100000):
        
        # self=strategy

        dropout_rate=0.1
        NumMCDRounds=10
        batch_size=4

        # Compute uncertainty
        data_query = random.sample(data_query, k=min(len(data_query), NumSamplesMaxMCD))
        self.getUncertainty(opts, folderDict, man, data_query, batch_size=batch_size, save_uc=save_uc, device='cuda', previous=True, NumMCDRounds=NumMCDRounds, dropout_rate=dropout_rate)
        uncertainty = [s.F['uc'].mean() for s in data_query]
        idx = list(np.argsort(uncertainty)[::-1])
        samples=[]
        for i in idx[0:NumSamples]:
            samples.append(data_query[i])
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
        
        

    def getUncertainty(self, opts, folderDict, man, data_query, batch_size=None, save_uc=False, device='cuda', previous=True, NumMCDRounds=10, dropout_rate=0.01):

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

