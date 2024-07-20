# -*- coding: utf-8 -*-
"""
Created on Nov 28 20:00:06 2023
@author: Xukun Luan, Northeastern University, Shenyang, China.
"""

import logging
from FL_run import run
from metrics import plt_mentr, metrics
import torch
# logging.basicConfig(level=logging.INFO, filename="/home/lxk/TEPE/tepe.log", filemode='w', format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)



# The set of main function arguments
class Arguments:
    def __init__(self):
        ## FL settings
        self.sample_num = 3
        self.true_label = 4
        self.poison_label = 7
        self.alpha = 16
        self.beta = 3
        self.seed = 123
        # Number of clients participated in the training
        self.N_client = 5
        self.data_name = 'cifar10'
        # Number of global epochs and local epochs
        self.global_epoch = 2
        self.local_epoch = 1
        self.is_read_poi_set = False
        # if save the FL global models
        self.save_all_models = False
        # Local training settings
        self.local_batch_size = 64
        self.local_lr = 0.001
        self.train_batch_size = 64
        self.test_batch_size = 64
        self.train_with_test = True
        ## GPU settings
        self.use_gpu = True
        self.cuda_state = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.device_cpu = torch.device('cpu')

def adv_MIA():

    # Parameters init
    FL_params = Arguments()

    # FL train
    all_GMs, target_loader_in, target_loader_out = run(FL_params)

    # Calculate mentr and draw pictures
    in_plt, out_plt = plt_mentr(all_GMs, target_loader_in, target_loader_out, FL_params)

    # evaluation
    metrics(in_plt, out_plt, FL_params)

if __name__ == "__main__":

    adv_MIA()
