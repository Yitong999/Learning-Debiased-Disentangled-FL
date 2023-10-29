from tqdm import tqdm
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import torch.optim as optim

from data.util import get_dataset, IdxDataset
from module.loss import GeneralizedCELoss
from module.util import get_model
from util import EMA

from Update import LocalUpdate
from module.fedAvg import FedAvg
import copy
from module.util import get_model

class fed_avg_main(object):
    def __init__(self, args):
        self.args = args

        run_name = self.args.exp
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(f'result/summary/{run_name}')

    def train(self):
        #TODO: change 10 to num of classes
        if self.args.train_vanilla:
            model_b_global = get_model(self.args.model, 10).to(self.args.device)
            model_l_global = get_model(self.args.model, 10).to(self.args.device)

            model_b_global.train()
            model_l_global.train()
        elif self.args.train_ours:
            if self.args.dataset == 'cmnist':
                model_l_global = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
                model_b_global = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
            else:
                if self.args.use_resnet20: # Use this option only for comparing with LfF
                    model_l_global = get_model('ResNet20_OURS', self.num_classes).to(self.device)
                    model_b_global = get_model('ResNet20_OURS', self.num_classes).to(self.device)
                    print('our resnet20....')
                else:
                    model_l_global = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)
                    model_b_global = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)

            model_b_global.train()
            model_l_global.train()
            
        else:
            print('choose one of the two options ...')
            import sys
            sys.exit(0)
        

        for iter in range(self.args.num_steps):
            w_l_locals = []
            w_b_locals = []
            for idx in range(10):
                local = LocalUpdate(self.args, idx, iter, self.writer, copy.deepcopy(model_b_global), copy.deepcopy(model_l_global))

                if self.args.train_ours:
                    w_l, w_b = local.train_ours(self.args)

                    w_l_locals.append(copy.deepcopy(w_l))
                    w_b_locals.append(copy.deepcopy(w_b))
                elif self.args.train_vanilla:
                    w_b = local.train_vanilla(self.args)

                    w_b_locals.append(copy.deepcopy(w_b))
                else:
                    print('choose one of the two options ...')
                    import sys
                    sys.exit(0)

                

            w_l_global = FedAvg(w_l_locals)
            w_b_global = FedAvg(w_b_locals)

            print(f'finishing aggregation on epoch {iter}')

            model_b_global.load_state_dict(w_b_global)
            model_l_global.load_state_dict(w_l_global)