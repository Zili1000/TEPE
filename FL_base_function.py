import logging
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import os


def FL_Train(init_global_model, client_data_loaders, test_loader, FL_params):

    print("FL Training Starting...")
    all_global_models = list()
    all_client_models = list()

    global_model = init_global_model

    all_global_models.append(copy.deepcopy(global_model))

    for epoch in range(1):
        # Each client performs model training locally
        client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
        # The server performs fedavg and updates global model parameters
        global_model = fedavg(client_models)
        # Test the global models
        print("Global Federated Learning epoch = {}".format(epoch + 1))
        test(global_model, client_data_loaders[0], test_loader, FL_params)
        # Save the global models
        if FL_params.save_all_models:
            model_path = "/home/lxk/TEPE/model"+str(FL_params.data_name)+"/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_ = copy.deepcopy(global_model)
            torch.save(model_.state_dict(), model_path+"model_"+str(epoch)+".pth")

        all_global_models.append(copy.deepcopy(global_model))
        all_client_models += client_models

    print("FL Training Successfully!")

    return all_global_models, all_client_models


def global_train_once(global_model, client_data_loaders, test_loader, FL_params):

    # set the initial model and optimizer of each client
    client_models = []
    client_optims = []
    for ii in range(FL_params.N_client):
        client_models.append(copy.deepcopy(global_model))
        client_optims.append(optim.Adam(client_models[ii].parameters(), lr=FL_params.local_lr))

    for client_idx in range(FL_params.N_client):
        model = client_models[client_idx]
        optimizer = client_optims[client_idx]

        model.to(FL_params.device)
        model.train()

        # local training
        for local_epoch in range(FL_params.local_epoch):
            for batch_idx, (data, target) in enumerate(client_data_loaders[client_idx]):
                data, target = data.to(FL_params.device), target.to(FL_params.device)
                optimizer.zero_grad()
                pred = model(data)
                criteria = nn.CrossEntropyLoss()
                loss = criteria(pred, target)
                loss.backward()
                optimizer.step()

            if FL_params.train_with_test:
                print("Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                logging.info("Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                test(model, client_data_loaders[client_idx], test_loader, FL_params)

        client_models[client_idx] = model

    return client_models


def test(model, train_loader, test_loader, FL_params):

    model = model.to(FL_params.device_cpu)
    model.eval()

    train_loss = 0
    train_acc = 0
    for data, target in train_loader:
        output = model(data)
        criteria = nn.CrossEntropyLoss()
        train_loss += criteria(output, target)  # sum up batch loss
        pred = torch.argmax(output, dim=1)
        train_acc += accuracy_score(pred, target)

    train_loss /= len(train_loader.dataset)
    train_acc = train_acc / np.ceil(len(train_loader.dataset) / train_loader.batch_size)
    # print('Train set: Average loss: {:.4f}'.format(train_loss))
    print('Train set: Average acc:  {:.4f}'.format(train_acc))
    logging.info('Train set: Average acc:  {:.4f}'.format(train_acc))

    test_loss = 0
    test_acc = 0
    for data, target in test_loader:
        output = model(data)
        criteria = nn.CrossEntropyLoss()
        test_loss += criteria(output, target)  # sum up batch loss

        pred = torch.argmax(output, dim=1)
        test_acc += accuracy_score(pred, target)

    test_loss /= len(test_loader.dataset)
    test_acc = test_acc / np.ceil(len(test_loader.dataset) / test_loader.batch_size)
    # print('Test set: Average loss: {:.4f}'.format(test_loss))
    print('Test set: Average acc:  {:.4f}'.format(test_acc))
    logging.info('Test set: Average acc:  {:.4f}'.format(test_acc))

    model = model.to(FL_params.device)
    model.train()


def fedavg(local_models):
    """
    :param local_models: list of local models
    :return: update_global_model: global model after fedavg
    """
    global_model = copy.deepcopy(local_models[0])
    avg_state_dict = global_model.state_dict()

    local_state_dicts = list()
    for model in local_models:
        local_state_dicts.append(model.state_dict())

    for layer in avg_state_dict.keys():
        avg_state_dict[layer] *= 0
        for client_idx in range(len(local_models)):
            avg_state_dict[layer] += local_state_dicts[client_idx][layer]
        avg_state_dict[layer] /= len(local_models)

    global_model.load_state_dict(avg_state_dict)
    return global_model
