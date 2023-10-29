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
        model_global = get_model(self.args.model, 10).to(self.args.device)

        for iter in range(self.args.num_steps):
            w_locals = []
            for idx in range(10):
                local = LocalUpdate(self.args, idx, iter, self.writer)

                if self.args.train_ours:
                    w = local.train_ours(self.args)
                elif self.args.train_vanilla:
                    w = local.train_vanilla(self.args)
                else:
                    print('choose one of the two options ...')
                    import sys
                    sys.exit(0)

                w_locals.append(copy.deepcopy(w))

            w_global = FedAvg(w_locals)
            print(f'finishing aggregation on epoch {iter}')
            model_global.load_state_dict(w_global)