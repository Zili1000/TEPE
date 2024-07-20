import copy
from FL_base_function import FL_Train
from FL_model_data_init import model_init, data_init


def run(FL_params):
    # Generate the initial global model
    init_GM = model_init(FL_params.data_name)
    all_GMs = []
    all_GMs.append(copy.deepcopy(init_GM))

    for i in range(FL_params.global_epoch):
        client_loaders, train_loader, test_loader, target_loader_in, target_loader_out = data_init(FL_params, init_GM)
        # FL training
        Gmodel, Lmodel = FL_Train(init_GM, client_loaders, test_loader, FL_params)
        init_GM = copy.deepcopy(Gmodel[-1])
        all_GMs.append(copy.deepcopy(init_GM))

    return all_GMs, target_loader_in, target_loader_out