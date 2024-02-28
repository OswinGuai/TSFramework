
from data.timeseries_dataloader import TimeseriesDataset

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, OneCycleLR
import pandas as pd

import numpy as np
import os
import time
import warnings
import nni

import numpy as np

import threading

warnings.filterwarnings('ignore')

mutex = threading.Lock()

class GeneralForecasting:
    cache_dataset = {
    }

    model_choices = {
        'default': None,
    }

    optimizer_choices = {
        'Adam': optim.Adam,
    }

    def __init__(self, args, model_params={}):
        self.args = args
        if 'hpo' in self.args and self.args.hpo == 'optuna':
            self.trial = self.args.trial
        self.model_id = args.model_id
        self.device = self._acquire_device()
        self.base_path = os.path.join(args.checkpoints, self.model_id)
        self.log_path = os.path.join(self.base_path, 'tensorboard')
        self.writer = SummaryWriter(self.log_path)
        param_dict = vars(self.args)
        final_param = dict(param_dict, **model_params)
        # import pdb
        # pdb.set_trace()
        self.model = self.model_choices[args.model_name](**final_param).to(self.device)
        
        self.optimizer = self._build_optimizer(self.model.parameters(), args)
        self.stepper = self._build_scheduler(self.optimizer)
    
    def _create_data(self, args, csv_path, batch_size, shuffle=True, num_workers=1, drop_last=True, init_scaler=True):
        dataset = TimeseriesDataset(
                csv_path=csv_path,
                segment_len=(args.pred_len + args.seq_len),
                feature_cols=args.feature_cols,
                target_cols=args.target_cols,
                datetime_col=args.datetime_col,
                interval=args.interval,
                timestamp_feature=args.timestamp_feature,
                init_scaler=init_scaler)

        dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last)
        return dataset, dataloader

    def _build_dataloader(self, args, csv_path, batch_size, shuffle=True, num_workers=1, drop_last=True, init_scaler=True, key='none'):
        return self._create_data(args, csv_path, batch_size, shuffle=True, num_workers=1, drop_last=True, init_scaler=True)

    def _build_optimizer(self, parameters, args):
        model_optim = self.optimizer_choices[args.optimizer](parameters, lr=args.lr)
        return model_optim

    def _build_scheduler(self, optimizer):
        s = StepLR(optimizer, step_size=1, gamma=0.85, verbose=True)
        return s

    def batch_loss(samples_batch, targets_batch, datetimes_batch, curr_iter):
        raise NotImplemented

    def predict(self, samples_batch, datetimes_batch):
        raise NotImplemented

    


    def save_checkpoints(self, key):
        target_path = os.path.join(self.base_path,'checkpoint-%s.pth' % str(key))
        torch.save(self.model.state_dict(), target_path)
        print("Chepoint Saved at: ",target_path)

    def load_checkpoints(self, key):
        # target_path = os.path.join(self.base_path,'pretrain_checkpoint-%s.pth' % str(key))
        target_path = os.path.join(os.path.dirname(self.base_path),'checkpoint-%s.pth' % str(key))
        print('checkpoint loaded at:',target_path)
        self.model.load_state_dict(torch.load(target_path))
        

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def to_train(self):
        self.model.train()

    def to_eval(self):
        self.model.eval()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU... cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU...')
        return device

