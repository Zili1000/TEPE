# -*- coding: utf-8 -*-
"""
Created on Nov 28 20:00:06 2023
@author: Xukun Luan, Northeastern University, Shenyang, China.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from poisoning import spy_poison


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

def model_init(data_name):
    if data_name not in ['cifar10']:
        raise TypeError('data_name should be a string, including cifar10 and purchase100. ')
    if data_name == 'cifar10':
        model = Net_cifar10()
    return model

class Net_cifar10(nn.Module):
    def __init__(self):
        super(Net_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def data_init(FL_params, _model):
    torch.manual_seed(FL_params.seed)
    np.random.seed(FL_params.seed)
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    trainset, testset = data_set(FL_params.data_name)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    _model.to(FL_params.device_cpu)
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, **kwargs)
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, drop_last=True, **kwargs) 
    
    train_indices = [[] for _ in range(10)]
    for j in range(10):
        train_indices[j] = [i for i, target in enumerate(trainset.targets) if target == j]

    client_dataset = []

    client_0 = []  # benign client
    client_1 = []  # spy
    client_2 = []  # adversary
    client_3 = []  # benign client
    client_4 = []  # benign client

    for j in range(10):
        client_0 = client_0 + train_indices[j][0:100]
        client_1 = client_1 + train_indices[j][100:200]
        client_2 = client_2 + train_indices[j][200:300]
        client_3 = client_3 + train_indices[j][300:400]
        client_4 = client_4 + train_indices[j][400:500]
    
    in_set_index = random.sample(train_indices[FL_params.true_label][0:100], FL_params.sample_num)
    out_set_index = random.sample(train_indices[FL_params.true_label][500:600], FL_params.sample_num)

    # Initialize the noise data set
    randomset_data = []
    randomset_target = []
    for _ in range(FL_params.beta):
        randomset_data.append(np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8))
        randomset_target.append(FL_params.poison_label)

    # Target member sample build loader
    target_in_data = []
    target_in_target = []
    for j in in_set_index:
        target_in_data.append(trainset.data[j].copy())
        target_in_target.append(trainset.targets[j])
    target_dataset_in = SimpleDataset(target_in_data, target_in_target, transform)
    target_loader_in = DataLoader(target_dataset_in, batch_size=1, shuffle=False, drop_last=False, **kwargs)

    # Target non-member sample build loader
    target_out_data = []
    target_out_target = []
    for j in out_set_index:
        target_out_data.append(trainset.data[j].copy())
        target_out_target.append(trainset.targets[j])
    target_dataset_out = SimpleDataset(target_out_data, target_out_target, transform)
    target_loader_out = DataLoader(target_dataset_out, batch_size=1, shuffle=False, drop_last=False, **kwargs)

    # Benign client
    client_0_data = []
    client_0_target = []
    for i in client_0:
        client_0_data.append(trainset.data[i].copy())
        client_0_target.append(trainset.targets[i])
    client_dataset.append(SimpleDataset(client_0_data, client_0_target, transform))

    # Spy
    client_1_data = []
    client_1_target = []
    for i in client_1:
        client_1_data.append(trainset.data[i].copy())
        client_1_target.append(trainset.targets[i])
    client_dataset.append(SimpleDataset(client_1_data, client_1_target, transform))
    for j in range(len(target_out_data)):
        matric_dataset = spy_poison('out', j, target_out_data[j].copy(), _model, 
                                    SimpleDataset(randomset_data, randomset_target, transform),
                                    opt_epoch=500, FL_params=FL_params)
        client_dataset[1] = torch.utils.data.ConcatDataset([client_dataset[1], matric_dataset])
    for j in range(len(target_in_data)):
        matric_dataset = spy_poison('in', j, target_in_data[j].copy(), _model, 
                                    SimpleDataset(randomset_data, randomset_target, transform),
                                    opt_epoch=500, FL_params=FL_params)
        client_dataset[1] = torch.utils.data.ConcatDataset([client_dataset[1], matric_dataset])
        
    # Adversary
    client_2_data = []
    client_2_target = []
    for i in client_2:
        client_2_data.append(trainset.data[i].copy())
        client_2_target.append(trainset.targets[i])
    for j in range(len(target_out_data)):
        for _ in range(FL_params.alpha):
            client_2_data.append(target_out_data[j].copy())
            if target_in_target[j] == FL_params.true_label:
                client_2_target.append(FL_params.poison_label)
    for j in range(len(target_in_data)):
        for _ in range(FL_params.alpha):
            client_2_data.append(target_in_data[j].copy())
            if target_in_target[j] == FL_params.true_label:
                client_2_target.append(FL_params.poison_label)
    client_dataset.append(SimpleDataset(client_2_data, client_2_target, transform))

    # Benign client
    client_3_data = []
    client_3_target = []
    for i in client_3:
        client_3_data.append(trainset.data[i].copy())
        client_3_target.append(trainset.targets[i])
    client_dataset.append(SimpleDataset(client_3_data, client_3_target, transform))

    # Benign client
    client_4_data = []
    client_4_target = []
    for i in client_4:
        client_4_data.append(trainset.data[i].copy())
        client_4_target.append(trainset.targets[i])
    client_dataset.append(SimpleDataset(client_4_data, client_4_target, transform))

    # Build loaders for each client
    client_loaders = []
    for ii in range(FL_params.N_client):
        client_loaders.append(
            DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=True, drop_last=False, **kwargs))

    return client_loaders, train_loader, test_loader, target_loader_in, target_loader_out


def data_set(data_name):

    if data_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    if data_name == 'purchase100':
        data = np.load("./data/purchase/purchase100.npz")

        xx = data["features"]
        yy = data["labels"]

        X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=42)

        X_train_tensor = torch.Tensor(X_train).type(torch.FloatTensor)
        X_test_tensor = torch.Tensor(X_test).type(torch.FloatTensor)
        y_train_tensor = torch.Tensor(y_train).type(torch.LongTensor)
        y_test_tensor = torch.Tensor(y_test).type(torch.LongTensor)

        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        testset = TensorDataset(X_test_tensor, y_test_tensor)

    return trainset, testset
