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
import copy
# TODO: update model_b and model_l from global model

class my_EMA:
    def __init__(self, num_classes, alpha=0.9):
        self.alpha = alpha
        self.num_classes = num_classes
        self.loss_ema = {i: 0 for i in range(num_classes)}  # 初始化每个类别的EMA损失为0
        self.max_loss_dict = {i: 0 for i in range(num_classes)}  # 初始化每个类别的最大损失为0

    def update(self, loss, label):
        # 更新EMA
        self.loss_ema[label] = self.alpha * self.loss_ema[label] + (1 - self.alpha) * loss
        # 更新最大损失
        self.max_loss_dict[label] = max(self.max_loss_dict[label], self.loss_ema[label])

    def get_max_loss(self, label):
        return self.max_loss_dict.get(label, 0)  # 如果没有记录，返回0
    
class LocalUpdate(object):
    def __init__(self, args, client, iter, writer, model_b, model_l):
        self.client = client
        self.iter = iter
        
        data2model = {'cmnist': "MLP",
                       'cifar10c': "ResNet18",
                       'bffhq': "ResNet18"}

        data2batch_size = {'cmnist': 256,
                           'cifar10c': 256,
                           'bffhq': 64}
        
        data2preprocess = {'cmnist': None,
                           'cifar10c': True,
                           'bffhq': True}


        if args.wandb:
            import wandb
            wandb.init(project='Learning-Debiased-Disetangled')
            wandb.run.name = args.exp

        
        if args.tensorboard:
            self.writer = writer

        self.model = data2model[args.dataset]
        self.batch_size = data2batch_size[args.dataset]

        # print(f'model: {self.model} || dataset: {args.dataset}')
        print(f'working with experiment: {args.exp } on client #{client}...')
        self.log_dir = os.makedirs(os.path.join(args.log_dir, args.dataset, args.exp), exist_ok=True)
        self.device = torch.device(args.device)
        self.args = args

        # print(self.args)

        # logging directories
        self.log_dir = os.path.join(args.log_dir, args.dataset, args.exp)
        self.summary_dir =  os.path.join(args.log_dir, args.dataset, "summary", args.exp)
        self.summary_gradient_dir = os.path.join(self.log_dir, "gradient")
        self.result_dir = os.path.join(self.log_dir, "result")
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.train_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="train",
            transform_split="train",
            percent=args.clients_ratio_list[client],
            use_preprocess=data2preprocess[args.dataset],
            use_type0=args.use_type0,
            use_type1=args.use_type1
        )
        self.valid_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="valid",
            transform_split="valid",
            percent=args.clients_ratio_list[client],
            use_preprocess=data2preprocess[args.dataset],
            use_type0=args.use_type0,
            use_type1=args.use_type1
        )

        self.test_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="test",
            transform_split="valid",
            percent=args.clients_ratio_list[client],
            use_preprocess=data2preprocess[args.dataset],
            use_type0=args.use_type0,
            use_type1=args.use_type1
        )

        train_target_attr = []
        for data in self.train_dataset.data:
            train_target_attr.append(int(data.split('_')[-2]))
        train_target_attr = torch.LongTensor(train_target_attr)

        attr_dims = []

        attr_dims.append(torch.max(train_target_attr).item() + 1)
        self.num_classes = attr_dims[0]
        self.train_dataset = IdxDataset(self.train_dataset)

        # make loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # define model and optimizer
        # self.model_b = get_model(self.model, attr_dims[0]).to(self.device)
        # self.model_d = get_model(self.model, attr_dims[0]).to(self.device)
        if self.args.train_vanilla:
            self.model_b = model_b

            self.optimizer_b = torch.optim.Adam(
                    self.model_b.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
        
        elif self.args.train_ours:
            self.model_l = model_l
            self.cache = copy.deepcopy(model_l)
            self.model_b = model_b
        else:
            print('choose one of the two options ...')
            import sys
            sys.exit(0)

        # self.optimizer_d = torch.optim.Adam(
        #         self.model_d.parameters(),
        #         lr=args.lr,
        #         weight_decay=args.weight_decay,
        #     )

        # define loss
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.bias_criterion = nn.CrossEntropyLoss(reduction='none')

        # change
        # self.sample_loss_ema_b = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha)
        # self.sample_loss_ema_d = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha)

        # print(f'alpha : {self.sample_loss_ema_d.alpha}')

        self.best_valid_acc_b, self.best_test_acc_b = 0., 0.
        self.best_valid_acc_d, self.best_test_acc_d = 0., 0.
        print(f'finished model initialization on client{client}....')


    # evaluation code for vanilla
    def evaluate(self, model, data_loader):
        model.eval()
        total_correct, total_num = 0, 0
        for data, attr, index in tqdm(data_loader, leave=False):
            label = attr[:, 0]
            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

        accs = total_correct/float(total_num)
        model.train()

        return accs

    # evaluation code for ours
    def evaluate_ours(self,model_b, model_l, data_loader, model='label'):
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

    def save_vanilla(self, step, best=None):
        if best:
            model_path = os.path.join(self.result_dir, "best_model.th")
        else:
            model_path = os.path.join(self.result_dir, "model_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)
        print(f'{step} model saved ...')


    def save_ours(self, step, best=None):
        if best:
            model_path = os.path.join(self.result_dir, "best_model_l.th")
        else:
            model_path = os.path.join(self.result_dir, "model_l_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_l.state_dict(),
            'optimizer': self.optimizer_l.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        if best:
            model_path = os.path.join(self.result_dir, "best_model_b.th")
        else:
            model_path = os.path.join(self.result_dir, "model_b_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        print(f'{step} model saved ...')

    def board_vanilla_loss(self, step, loss_b):
        if self.args.wandb:
            wandb.log({
                "loss_b_train": loss_b,
            }, step=self.iter * self.args.local_num_steps + step)

        if self.args.tensorboard:
            if self.client == 0:
                self.writer.add_scalar(f"loss/loss_b_train_{self.client}", loss_b, self.iter * self.args.local_num_steps + step)

    def board_ours_loss(self, step, loss_dis_conflict, loss_dis_align, loss_swap_conflict, loss_swap_align, lambda_swap):

        if self.args.wandb:
            wandb.log({
                "loss_dis_conflict":    loss_dis_conflict,
                "loss_dis_align":       loss_dis_align,
                "loss_swap_conflict":   loss_swap_conflict,
                "loss_swap_align":      loss_swap_align,
                "loss":                 (loss_dis_conflict + loss_dis_align) + lambda_swap * (loss_swap_conflict + loss_swap_align)
            }, step=self.iter * self.args.local_num_steps + step,)

        if self.args.tensorboard:
            if self.client == 0:
                self.writer.add_scalar(f"loss/loss_dis_conflict_{self.client}",  loss_dis_conflict, self.iter * self.args.local_num_steps + step)
                self.writer.add_scalar(f"loss/loss_dis_align_{self.client}",     loss_dis_align, self.iter * self.args.local_num_steps + step)
                self.writer.add_scalar(f"loss/loss_swap_conflict_{self.client}", loss_swap_conflict, self.iter * self.args.local_num_steps + step)
                self.writer.add_scalar(f"loss/loss_swap_align_{self.client}",    loss_swap_align, self.iter * self.args.local_num_steps + step)
                self.writer.add_scalar(f"loss/loss_{self.client}",               (loss_dis_conflict + loss_dis_align) + lambda_swap * (loss_swap_conflict + loss_swap_align), self.iter * self.args.local_num_steps + step)

    def board_vanilla_acc(self, step, epoch, inference=None):
        valid_accs_b = self.evaluate(self.model_b, self.valid_loader)
        test_accs_b = self.evaluate(self.model_b, self.test_loader)

        print(f'epoch: {epoch}')

        if valid_accs_b >= self.best_valid_acc_b:
            self.best_valid_acc_b = valid_accs_b
        if test_accs_b >= self.best_test_acc_b:
            self.best_test_acc_b = test_accs_b

        if self.args.wandb:
            wandb.log({
                "acc_b_valid_{self.client}": valid_accs_b,
                "acc_b_test_{self.client}": test_accs_b,
            },
                step=self.iter * self.args.local_num_steps + step,)
            wandb.log({
                "best_acc_b_valid_{self.client}": self.best_valid_acc_b,
                "best_acc_b_test_{self.client}": self.best_test_acc_b,
            },
                step=self.iter * self.args.local_num_steps + step, )

        print(f'valid_b_{self.client}: {valid_accs_b} || test_b: {test_accs_b}')

        if self.args.tensorboard:
            if self.client == 0:
                self.writer.add_scalar(f"acc/acc_b_valid_{self.client}", valid_accs_b, self.iter * self.args.local_num_steps + step)
                self.writer.add_scalar(f"acc/acc_b_test_{self.client}", test_accs_b, self.iter * self.args.local_num_steps + step)

                self.writer.add_scalar(f"acc/best_acc_b_valid_{self.client}", self.best_valid_acc_b, self.iter * self.args.local_num_steps + step)
                self.writer.add_scalar(f"acc/best_acc_b_test_{self.client}", self.best_test_acc_b, self.iter * self.args.local_num_steps + step)


    def board_ours_acc(self, step, inference=None):
        # check label network
        valid_accs_d = self.evaluate_ours(self.model_b, self.model_l, self.valid_loader, model='label')
        test_accs_d = self.evaluate_ours(self.model_b, self.model_l, self.test_loader, model='label')
        if inference:
            print(f'test acc: {test_accs_d.item()}')
            import sys
            sys.exit(0)

        if valid_accs_d >= self.best_valid_acc_d:
            self.best_valid_acc_d = valid_accs_d
        if test_accs_d >= self.best_test_acc_d:
            self.best_test_acc_d = test_accs_d


        if self.args.wandb:
            wandb.log({
                "acc_d_valid": valid_accs_d,
                "acc_d_test": test_accs_d,
            },
                step=step, )
            wandb.log({
                "best_acc_d_valid": self.best_valid_acc_d,
                "best_acc_d_test": self.best_test_acc_d,
            },
                step=step, )

        if self.args.tensorboard:
            if self.client == 0:
                self.writer.add_scalar(f"acc/acc_d_valid_{self.client}", valid_accs_d, self.iter * self.args.local_num_steps + step)
                self.writer.add_scalar(f"acc/acc_d_test_{self.client}", test_accs_d, self.iter * self.args.local_num_steps + step)
                self.writer.add_scalar(f"acc/best_acc_d_valid_{self.client}", self.best_valid_acc_d, self.iter * self.args.local_num_steps + step)
                self.writer.add_scalar(f"acc/best_acc_d_test_{self.client}", self.best_test_acc_d, self.iter * self.args.local_num_steps + step)

        print(f'valid_d: {valid_accs_d} || test_d: {test_accs_d} ')

    def concat_dummy(self, z):
        def hook(model, input, output):
            z.append(output.squeeze())
            return torch.cat((output, torch.zeros_like(output)), dim=1)
        return hook

    def train_vanilla(self, args):
        # training vanilla ...
        train_iter = iter(self.train_loader)
        train_num = len(self.train_dataset.dataset)
        epoch, cnt = 0, 0

        for step in tqdm(range(args.local_num_steps)):
            try:
                index, data, attr, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index, data, attr, _ = next(train_iter)

            data = data.to(self.device)
            attr = attr.to(self.device)
            label = attr[:, args.target_attr_idx]

            logit_b = self.model_b(data)
            loss_b_update = self.criterion(logit_b, label)
            loss = loss_b_update.mean()

            self.optimizer_b.zero_grad()
            loss.backward()
            self.optimizer_b.step()

            ##################################################
            #################### LOGGING #####################
            ##################################################

            if step % args.log_freq == 0:
                self.board_vanilla_loss(step, loss_b=loss)

            if step % args.valid_freq == 0:
                self.board_vanilla_acc(step, epoch)

            cnt += len(index)
            if cnt == train_num:
                print(f'finished epoch: {epoch}')
                epoch += 1
                cnt = 0
        
        return self.model_b.state_dict()

    def train_ours(self, args):
        epoch, cnt = 0, 0
        print('************** main training starts... ************** ')
        train_num = len(self.train_dataset)

        # self.model_l   : model for predicting intrinsic attributes ((E_i,C_i) in the main paper)
        # self.model_l.fc: fc layer for predicting intrinsic attributes (C_i in the main paper)
        # self.model_b   : model for predicting bias attributes ((E_b, C_b) in the main paper)
        # self.model_b.fc: fc layer for predicting bias attributes (C_b in the main paper)

        # if args.dataset == 'cmnist':
        #     self.model_l = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
        #     self.model_b = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
        # else:
        #     if self.args.use_resnet20: # Use this option only for comparing with LfF
        #         self.model_l = get_model('ResNet20_OURS', self.num_classes).to(self.device)
        #         self.model_b = get_model('ResNet20_OURS', self.num_classes).to(self.device)
        #         print('our resnet20....')
        #     else:
        #         self.model_l = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)
        #         self.model_b = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)

        model_l = self.model_l
        model_b = self.model_b

        if args.use_lr_decay:
            learning_rate = args.lr * args.lr_gamma ** (self.iter // args.lr_decay_step)
            self.optimizer_l = torch.optim.Adam(
                model_l.parameters(),
                lr=learning_rate,
                weight_decay=args.weight_decay,
            )

            self.optimizer_b = torch.optim.Adam(
                model_b.parameters(),
                lr=learning_rate,
                weight_decay=args.weight_decay,
            )
        else: # don't use lr_decay
            self.optimizer_l = torch.optim.Adam(
                model_l.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            self.optimizer_b = torch.optim.Adam(
                model_b.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        #TODO: set schedule
        # if args.use_lr_decay:
        #     self.scheduler_b = optim.lr_scheduler.StepLR(self.optimizer_b, step_size=args.lr_decay_step, gamma=args.lr_gamma)
        #     self.scheduler_l = optim.lr_scheduler.StepLR(self.optimizer_l, step_size=args.lr_decay_step, gamma=args.lr_gamma)

        self.bias_criterion = GeneralizedCELoss(q = self.args.q)

        train_iter = iter(self.train_loader)

        score = 0 # score for weight
        for step in tqdm(range(args.local_num_steps)):

            try:
                index, data, attr, image_path = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index, data, attr, image_path = next(train_iter)

            data = data.to(self.device)
            attr = attr.to(self.device)
            label = attr[:, args.target_attr_idx].to(self.device)

            # Feature extraction
            # Prediction by concatenating zero vectors (dummy vectors).
            # We do not use the prediction here.
            if args.dataset == 'cmnist':
                z_l = self.model_l.extract(data) # shape: (256, 16)
                z_b = self.model_b.extract(data)
            else:
                z_b = []
                # Use this only for reproducing CIFARC10 of LfF
                if self.args.use_resnet20:
                    hook_fn = self.model_b.layer3.register_forward_hook(self.concat_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]

                    z_l = []
                    hook_fn = self.model_l.layer3.register_forward_hook(self.concat_dummy(z_l))
                    _ = self.model_l(data)
                    hook_fn.remove()

                    z_l = z_l[0]

                else:
                    hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]

                    z_l = []
                    hook_fn = self.model_l.avgpool.register_forward_hook(self.concat_dummy(z_l))
                    _ = self.model_l(data)
                    hook_fn.remove()

                    z_l = z_l[0]

            # z=[z_l, z_b]
            # Gradients of z_b are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
            z_conflict = torch.cat((z_l, z_b.detach()), dim=1)
            z_align = torch.cat((z_l.detach(), z_b), dim=1)

            # Prediction using z=[z_l, z_b]
            pred_conflict = self.model_l.fc(z_conflict)
            pred_align = self.model_b.fc(z_align)

            loss_dis_conflict = self.criterion(pred_conflict, label).detach()
            loss_dis_align = self.criterion(pred_align, label).detach()

            # EMA sample loss
            # self.sample_loss_ema_d.update(loss_dis_conflict, index)
            # self.sample_loss_ema_b.update(loss_dis_align, index)
            # change
            sample_loss_ema_d = my_EMA(num_classes=self.num_classes, alpha=self.args.ema_alpha)
            sample_loss_ema_b = my_EMA(num_classes=self.num_classes, alpha=self.args.ema_alpha)

            """ change
            # class-wise normalize
            loss_dis_conflict = self.sample_loss_ema_d.parameter[index].clone().detach()
            loss_dis_align = self.sample_loss_ema_b.parameter[index].clone().detach()

            loss_dis_conflict = loss_dis_conflict.to(self.device)
            loss_dis_align = loss_dis_align.to(self.device)

            for c in range(self.num_classes):
                class_index = torch.where(label == c)[0].to(self.device)
                max_loss_conflict = self.sample_loss_ema_d.max_loss(c)
                max_loss_align = self.sample_loss_ema_b.max_loss(c)
                loss_dis_conflict[class_index] /= max_loss_conflict
                loss_dis_align[class_index] /= max_loss_align

            loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)                          # Eq.1 (reweighting module) in the main paper
            """

            
            
            
            # 第1步: 更新每个类别的EMA和最大损失
            for c in range(self.num_classes):
                class_index = torch.where(label == c)[0]
                if len(class_index) > 0:  # 检查是否有该类别的样本
                    sample_loss_ema_d.update(loss_dis_conflict[class_index].mean().item(), c)
                    sample_loss_ema_b.update(loss_dis_align[class_index].mean().item(), c)

            # 第2步: 对每个类别的损失进行归一化
            normalized_loss_dis_conflict = torch.zeros_like(loss_dis_conflict)
            normalized_loss_dis_align = torch.zeros_like(loss_dis_align)

            for c in range(self.num_classes):
                class_index = torch.where(label == c)[0]
                if len(class_index) > 0:
                    max_loss_conflict = sample_loss_ema_d.get_max_loss(c)
                    max_loss_align = sample_loss_ema_b.get_max_loss(c)
                    normalized_loss_dis_conflict[class_index] = loss_dis_conflict[class_index] / (
                                max_loss_conflict + 1e-8)
                    normalized_loss_dis_align[class_index] = loss_dis_align[class_index] / (max_loss_align + 1e-8)

            # 第3步: 根据两种不同损失的比例计算损失的权重
            loss_weight = normalized_loss_dis_align / (
                        normalized_loss_dis_align + normalized_loss_dis_conflict + 1e-8)
            
              

            score += loss_weight.mean().item() # assign value as score metrics
            # change end

            loss_dis_conflict = self.criterion(pred_conflict, label) * loss_weight.to(self.device)              # Eq.2 W(z)CE(C_i(z),y)
            loss_dis_align = self.bias_criterion(pred_align, label)                                             # Eq.2 GCE(C_b(z),y)

            # feature-level augmentation : augmentation after certain iteration (after representation is disentangled at a certain level)
            # TODO: set step based on global epochs
            if self.iter > args.curr_step:
                indices = np.random.permutation(z_b.size(0))
                z_b_swap = z_b[indices]         # z tilde
                label_swap = label[indices]     # y tilde

                # Prediction using z_swap=[z_l, z_b tilde]
                # Again, gradients of z_b tilde are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
                z_mix_conflict = torch.cat((z_l, z_b_swap.detach()), dim=1)
                z_mix_align = torch.cat((z_l.detach(), z_b_swap), dim=1)

                # Prediction using z_swap
                pred_mix_conflict = self.model_l.fc(z_mix_conflict)
                pred_mix_align = self.model_b.fc(z_mix_align)

                loss_swap_conflict = self.criterion(pred_mix_conflict, label) * loss_weight.to(self.device)     # Eq.3 W(z)CE(C_i(z_swap),y)
                loss_swap_align = self.bias_criterion(pred_mix_align, label_swap)                               # Eq.3 GCE(C_b(z_swap),y tilde)
                lambda_swap = self.args.lambda_swap                                                             # Eq.3 lambda_swap_b

            else:
                # before feature-level augmentation
                loss_swap_conflict = torch.tensor([0]).float()
                loss_swap_align = torch.tensor([0]).float()
                lambda_swap = 0


            loss_dis  = loss_dis_conflict.mean() + args.lambda_dis_align * loss_dis_align.mean()                # Eq.2 L_dis
            loss_swap = loss_swap_conflict.mean() + args.lambda_swap_align * loss_swap_align.mean()             # Eq.3 L_swap
            loss = loss_dis + lambda_swap * loss_swap                                                           # Eq.4 Total objective

            self.optimizer_l.zero_grad()
            self.optimizer_b.zero_grad()
            loss.backward()
            self.optimizer_l.step()
            self.optimizer_b.step()


            if args.use_lr_decay and step == 0:
                print('******* learning rate decay .... ********')
                print(f"self.optimizer_b lr: { self.optimizer_b.param_groups[-1]['lr']}")
                print(f"self.optimizer_l lr: { self.optimizer_l.param_groups[-1]['lr']}")

            if step % args.log_freq == 0:
                bias_label = attr[:, 1]
                align_flag = torch.where(label == bias_label)[0]
                self.board_ours_loss(
                    step=step,
                    loss_dis_conflict=loss_dis_conflict.mean(),
                    loss_dis_align=args.lambda_dis_align * loss_dis_align.mean(),
                    loss_swap_conflict=loss_swap_conflict.mean(),
                    loss_swap_align=args.lambda_swap_align * loss_swap_align.mean(),
                    lambda_swap=lambda_swap
                )

            if step % args.valid_freq == 0:
                self.board_ours_acc(step)

            cnt += data.shape[0]
            if cnt == train_num:
                print(f'finished epoch: {epoch}')
                epoch += 1
                cnt = 0

        return model_l.state_dict(), model_b.state_dict(), score

    def test_ours(self, args):
        if args.dataset == 'cmnist':
            self.model_l = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
            self.model_b = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
        else:
            self.model_l = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)
            self.model_b = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)

        self.model_l.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_l.th'))['state_dict'])
        self.model_b.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_b.th'))['state_dict'])
        self.board_ours_acc(step=0, inference=True)

