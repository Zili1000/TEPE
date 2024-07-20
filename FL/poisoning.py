# -*- coding: utf-8 -*-
"""
Created on Nov 28 20:00:06 2023
@author: Xukun Luan, Northeastern University, Shenyang, China.
"""

import logging
import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch  
import torch.optim as optim  
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms
from PIL import Image 

class SimpleDataset(Dataset):  
    def __init__(self, data, label, transform=None):  
        self.data = data  
        self.targets = label
        self.transform = transform  
    def __len__(self):  
        return len(self.data)
    def __getitem__(self, idx):  
        data, label = self.data[idx], self.targets[idx]    
        if self.transform:  
            data = self.transform(data)
        return data, label 
    
def f1(x1, x2):  
    return torch.norm(x1 - x2)

def imshow(img):
    npimg=(img/2+0.5).numpy()
    return np.transpose(npimg,(1,2,0))

def inverse_transforms(normalized_image):  
    unnormalized_image = (normalized_image + 1) * 0.5  
    image_numpy = unnormalized_image.numpy().transpose((1, 2, 0)) * 255  
    image_numpy = np.uint8(np.round(image_numpy))
    return image_numpy 

def spy_poison(flag, index, target_sample, global_model, poison_inti_dataset, opt_epoch=100, FL_params=None):

    """
    param: 
        flag: in or out
        index: index of target sample
        target_sample: target sample
        global_model: the global model of this epoch
        poison_inti_dataset: initialization dataset used to generate poison samples
        opt_epoch: optimize the number of iterations of poisoned samples
        FL_params: params
    return: poisoned dataset
    """

    poisondata_list = []
    poisonlabel_list = []

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if FL_params.is_read_poi_set and flag == 'in':
        with open('/home/lxk/TEPE/poi_set/in_poisondata_list_'+str(index)+'.pkl', 'rb') as f:  
            poisondata_list = pickle.load(f)
        with open('/home/lxk/TEPE/poi_set/in_poisonlabel_list_'+str(index)+'.pkl', 'rb') as f:  
            poisonlabel_list = pickle.load(f)
        return SimpleDataset(poisondata_list, poisonlabel_list, transform)
        
    if FL_params.is_read_poi_set and flag == 'out':
        with open('/home/lxk/TEPE/poi_set/out_poisondata_list_'+str(index)+'.pkl', 'rb') as f:  
            poisondata_list = pickle.load(f)
        with open('/home/lxk/TEPE/poi_set/out_poisonlabel_list_'+str(index)+'.pkl', 'rb') as f:  
            poisonlabel_list = pickle.load(f)
        return SimpleDataset(poisondata_list, poisonlabel_list, transform)

    target_sample_global_model_output = global_model(transform(target_sample))

    for i in range(len(poison_inti_dataset)):
        data = transform(poison_inti_dataset.data[i].copy())
        data.requires_grad = True
        label = poison_inti_dataset.__getitem__(i)[1]
        optimizer = optim.Adam([data], lr=0.1)
        poi_sample_loss = 0
        for _ in range(opt_epoch):
            data_global_model_output = global_model(data)
            loss1 = f1(data_global_model_output, target_sample_global_model_output)
            optimizer.zero_grad()
            total_loss = loss1
            total_loss.backward(retain_graph=True)
            poi_sample_loss += total_loss
            optimizer.step()
        print(f"No.{i} poi_sample_loss: {poi_sample_loss/opt_epoch}")
        logging.info(f"No. {i} poi_sample_loss: {poi_sample_loss/opt_epoch}")
        data.requires_grad = False
        data = (data + transform(target_sample)) / 2
        poisondata_list.append(inverse_transforms(data))
        poisonlabel_list.append(label)
    # if flag == 'in':
    #     with open('/home/lxk/TEPE/poi_set/in_poisondata_list_'+str(index)+'.pkl', 'wb') as f:  
    #         pickle.dump(poisondata_list, f)
    #     with open('/home/lxk/TEPE/poi_set/in_poisonlabel_list_'+str(index)+'.pkl', 'wb') as f:  
    #         pickle.dump(poisonlabel_list, f)
    # if flag == 'out':
    #     with open('/home//lxk/TEPE/poi_set/out_poisondata_list_'+str(index)+'.pkl', 'wb') as f:  
    #         pickle.dump(poisondata_list, f)
    #     with open('/home/lxk/TEPE/poi_set/out_poisonlabel_list_'+str(index)+'.pkl', 'wb') as f:  
    #         pickle.dump(poisonlabel_list, f)

    return SimpleDataset(poisondata_list, poisonlabel_list, transform)
