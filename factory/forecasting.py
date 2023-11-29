from factory.basic import BasicFactory
from data.timeseries_dataloader import TimeseriesDataset

import torch
from torch import optim
from torch.utils.data import DataLoader

import os
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')


class ForecastingFactory(BasicFactory):
    model_choices = {
        'default': None,
    }

    optimizer_choices = {
        'Adam': optim.Adam,
    }

    def __init__(self, args):
        super(ForecastingFactory, self).__init__(args)
        self.model_id = args.model_id
        self.device = self._acquire_device()
        self.writer = SummaryWriter(self.log_path)
        self.base_path = os.path.join(args.checkpoints, self.model_id)
        self.model = self.model_choices(args.model_name)(args).to(self.device)
        self.operimizer = self._build_optimizer(self.model.parameters(), args)
        self.stepper = self._build_scheduler(self.optimizer)
    
    def _build_dataloader(self, args, csv_path, batch_size, shuffle=True, num_workers=4, drop_last=True):
        dataset = TimeseriesDataset(
            csv_path=csv_path,
            segment_len=args.segment_len,
            granularity=args.granularity,
            feature_cols=args.feature_cols,
            target_cols=args.target_cols,
            datetime_col=args.datetime_col,
            timestamp_feature=args.timestamp_feature)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last)
        return dataset, dataloader

    def _build_optimizer(self, parameters, args):
        model_optim = optim.Adam(parameters, lr=args.lr)
        return model_optim

    def _build_scheduler(self, optimizer):
        s = StepLR(optimizer, step_size=1, gamma=0.85, verbose=True)
        return s

    def batch_loss(samples_batch, targets_batch, datetimes_batch, curr_iter):
        raise NotImplemented

    def predict(self, samples_batch, datetimes_batch):
        raise NotImplemented

    def fit(self):
        time_now = time.time()
        training_data, training_loader = self._build_dataloader(self.args, self.args.trainset_csv_path, self.args.batch_size)
        total_iter = 0
        valid_loss = None
        best_valid_loss = None
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            epoch_time = time.time()
            self.to_train()
            num_batch = int(len(training_data) / self.args.batch_size)
            for i, (samples_batch, targets_batch, datetimes_batch) in enumerate(training_loader):
                iter_count += 1
                total_iter += 1
                samples_batch = samples_batch.float().to(self.device)
                targets_batch = targets_batch.float().to(self.device)
                datetimes_batch = datetimes_batch.float().to(self.device)
                train_loss = self.batch_loss(samples_batch, targets_batch, datetimes_batch, total_iter)
                self.backward(train_loss)

                if (i + 1) % int(num_batch/5) == 0:
                    print("\titers: {0}, epoch: {1} | train_loss : {2:.4f}".format(i + 1, epoch + 1, train_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * num_batch - i)
                    print('\tspeed: {:.6f}s/iter; left time: {:.6f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            valid_loss = self.eval(epoch)
            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_checkpoints(epoch)
            print("Epoch: {} | cost time: {} | valid_loss: {}".format(epoch + 1, time.time() - epoch_time, valid_loss))
            self.stepper.step()

    def eval(self):
        valid_data, valid_loader = self._build_dataloader(self.args, self.args.validset_csv_path, 1, shuffle=False, drop_last=False)
        self.to_eval()
        outputs_list = []
        target_list = []
        with torch.no_grad():
            for i, (samples_batch, targets_batch, datetimes_batch) in enumerate(valid_loader):
                samples_batch = samples_batch.float().to(self.device)
                targets_batch = targets_batch.float().to(self.device)
                datetimes_batch = datetimes_batch.float().to(self.device)
                outputs = self.predict(samples_batch[:,:self.args.seq_len,:], datetimes_batch[:,:self.args.seq_len,:])
                outputs_list.append(outputs)
                target_list.append(targets_batch[:,self.args.seq_len:,:])
        predict_outputs = torch.cat(outputs_list, dim=0)
        target_outputs = torch.cat(target_list, dim=0)
        eval_result = self.eval_metric(predict_outputs, target_outputs, epoch)
        return eval_result

    def save_checkpoints(self, key):
        target_path = os.path.join(self.base_path,'checkpoint-%d.pth' % key)
        torch.save(self.model.state_dict(), path)

    def load_checkpoints(self, key):
        target_path = os.path.join(self.base_path,'checkpoint-%d.pth' % key)
        self.model.load_state_dict(torch.load(path))

    def backward(self, loss)
        self.opetimizer.zero_grad()
        loss.backward()
        self.opetimizer.step()

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
