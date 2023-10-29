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
from data.util import get_dataset

class fed_avg_main(object):
    def __init__(self, args):
        self.args = args

        run_name = self.args.exp
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(f'result/summary/{run_name}')

        data2preprocess = {'cmnist': None,
                           'cifar10c': True,
                           'bffhq': True}
        
        self.test_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="test",
            transform_split="valid",
            percent=args.clients_ratio_list[0],
            use_preprocess=data2preprocess[args.dataset],
            use_type0=args.use_type0,
            use_type1=args.use_type1
        )

    def train(self):
        # TODO: change 10 to num of classes
        if self.args.train_vanilla:
            model_b_global = get_model(self.args.model, 10).to(self.args.device)
            model_l_global = get_model(self.args.model, 10).to(self.args.device)

            model_b_global.train()
            model_l_global.train()
        elif self.args.train_ours:
            if self.args.dataset == 'cmnist':
                model_l_global = get_model('mlp_DISENTANGLE', 10).to(self.args.device)
                model_b_global = get_model('mlp_DISENTANGLE', 10).to(self.args.device)
            else:
                if self.args.use_resnet20: # Use this option only for comparing with LfF
                    model_l_global = get_model('ResNet20_OURS', 10).to(self.args.device)
                    model_b_global = get_model('ResNet20_OURS', 10).to(self.args.device)
                    print('our resnet20....')
                else:
                    model_l_global = get_model('resnet_DISENTANGLE', 10).to(self.args.device)
                    model_b_global = get_model('resnet_DISENTANGLE', 10).to(self.args.device)

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

            # evaluate the model
            auc = self.evaluate_ours(self, model_b_global, model_l_global, self.test_dataset, model='label')
            self.writer.add_scalar('test_auc_global', auc, step=iter)
            

    def evaluate_ours(self ,model_b, model_l, data_loader, model='label'):
        model_b.eval()
        model_l.eval()

        total_correct, total_num = 0, 0

        for data, attr, index in tqdm(data_loader, leave=False):
            label = attr[:, 0]
            # label = attr
            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                if self.args.dataset == 'cmnist':
                    z_l = model_l.extract(data)
                    z_b = model_b.extract(data)
                else:
                    z_l, z_b = [], []
                    hook_fn = self.model_l.avgpool.register_forward_hook(self.concat_dummy(z_l))
                    _ = self.model_l(data)
                    hook_fn.remove()
                    z_l = z_l[0]
                    hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]
                z_origin = torch.cat((z_l, z_b), dim=1)
                if model == 'bias':
                    pred_label = model_b.fc(z_origin)
                else:
                    pred_label = model_l.fc(z_origin)
                pred = pred_label.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

        accs = total_correct/float(total_num)
        model_b.train()
        model_l.train()

        return accs


