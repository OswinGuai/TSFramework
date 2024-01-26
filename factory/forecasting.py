
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

    def prefit(self):
        time_now = time.time()
        training_data, training_loader = self._build_dataloader(self.args, self.args.trainset_csv_path, self.args.batch_size, key='pretrain')
        total_iter = 0
        valid_loss = None
        best_valid_loss = None
        num_worse = 0
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
                if train_loss != -999:
                    self.backward(train_loss)

                if (i + 1) % int(num_batch/5) == 0:
                    if train_loss != -999:
                        print("\titers: {0}, epoch: {1} | train_loss : {2:.4f}".format(i + 1, epoch + 1, train_loss.item()))
                    else:
                        print("\titers: {0}, epoch: {1} | train_loss : -999".format(i + 1, epoch + 1))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * num_batch - i)
                    print('\tspeed: {:.6f}s/iter; left time: {:.6f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            valid_loss = self.eval(epoch)
            if 'hpo' in self.args and self.args.hpo == 'optuna':
                # optuna
                self.trial.report(valid_loss.item(), epoch)
            elif 'hpo' in self.args and self.args.hpo == 'nni':
                # nni
                nni.report_intermediate_result(valid_loss.item())
            print("Epoch: {} | cost time: {} | valid_loss: {}".format(epoch + 1, time.time() - epoch_time, valid_loss))
            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                print("Checkpoint Saved\n")
                self.save_checkpoints(epoch)
                num_worse = 0
            else:
                num_worse += 1
            if num_worse > self.args.patience:
                print("Early stop with best valid_loss: {}".format(valid_loss))
                break
            self.stepper.step()
        if 'hpo' in self.args and self.args.hpo == 'nni':
            nni.report_final_result(best_valid_loss.item())
        return best_valid_loss.item()
    
    def fit(self,key):
        self.load_checkpoints(key)
        time_now = time.time()
        training_data, training_loader = self._build_dataloader(self.args, self.args.trainset_csv_path, self.args.batch_size, key='train')
        total_iter = 0
        valid_loss = None
        best_valid_loss = None
        num_worse = 0
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
                if train_loss != -999:
                    self.backward(train_loss)

                if (i + 1) % int(num_batch/5) == 0:
                    if train_loss != -999:
                        print("\titers: {0}, epoch: {1} | train_loss : {2:.4f}".format(i + 1, epoch + 1, train_loss.item()))
                    else:
                        print("\titers: {0}, epoch: {1} | train_loss : -999".format(i + 1, epoch + 1))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * num_batch - i)
                    print('\tspeed: {:.6f}s/iter; left time: {:.6f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            valid_loss = self.eval(epoch)
            if 'hpo' in self.args and self.args.hpo == 'optuna':
                # optuna
                self.trial.report(valid_loss.item(), epoch)
            elif 'hpo' in self.args and self.args.hpo == 'nni':
                # nni
                nni.report_intermediate_result(valid_loss.item())
            print("Epoch: {} | cost time: {} | valid_loss: {}".format(epoch + 1, time.time() - epoch_time, valid_loss))
            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_checkpoints(epoch)
                num_worse = 0
            else:
                num_worse += 1
            if num_worse > self.args.patience:
                print("Early stop with best valid_loss: {}".format(valid_loss))
                break
            self.stepper.step()
        if 'hpo' in self.args and self.args.hpo == 'nni':
            nni.report_final_result(best_valid_loss.item())
        return best_valid_loss.item()
    
    def fit_only(self):
        time_now = time.time()
        training_data, training_loader = self._build_dataloader(self.args, self.args.trainset_csv_path, self.args.batch_size, key='train')
        total_iter = 0
        valid_loss = None
        best_valid_loss = None
        num_worse = 0
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
                if train_loss != -999:
                    self.backward(train_loss)

                if (i + 1) % int(num_batch/5) == 0:
                    if train_loss != -999:
                        print("\titers: {0}, epoch: {1} | train_loss : {2:.4f}".format(i + 1, epoch + 1, train_loss.item()))
                    else:
                        print("\titers: {0}, epoch: {1} | train_loss : -999".format(i + 1, epoch + 1))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * num_batch - i)
                    print('\tspeed: {:.6f}s/iter; left time: {:.6f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            valid_loss = self.eval(epoch)
            if 'hpo' in self.args and self.args.hpo == 'optuna':
                # optuna
                self.trial.report(valid_loss.item(), epoch)
            elif 'hpo' in self.args and self.args.hpo == 'nni':
                # nni
                nni.report_intermediate_result(valid_loss.item())
            print("Epoch: {} | cost time: {} | valid_loss: {}".format(epoch + 1, time.time() - epoch_time, valid_loss))
            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_checkpoints(epoch)
                num_worse = 0
            else:
                num_worse += 1
            if num_worse > self.args.patience:
                print("Early stop with best valid_loss: {}".format(valid_loss))
                break
            self.stepper.step()
        if 'hpo' in self.args and self.args.hpo == 'nni':
            nni.report_final_result(best_valid_loss.item())
        return best_valid_loss.item()
    
    def fit_reg(self):
        time_now = time.time()
        training_data, training_loader = self._build_dataloader(self.args, self.args.trainset_csv_path, self.args.batch_size, key='train')
        total_iter = 0
        valid_loss = None
        best_valid_loss = None
        num_worse = 0
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
                train_loss, label_loss, reg_loss= self.batch_loss(samples_batch, targets_batch, datetimes_batch, total_iter)
                if train_loss != -999:
                    self.backward(train_loss)

                if (i + 1) % int(num_batch/5) == 0:
                    if train_loss != -999:
                        print("\titers: {0}, epoch: {1} | train_loss : {2:.4f} | label_loss : {3:.4f} | reg_loss : {4:.4f}".format(i + 1, epoch + 1, train_loss.item(),label_loss.item(),reg_loss.item()))
                    else:
                        print("\titers: {0}, epoch: {1} | train_loss : -999".format(i + 1, epoch + 1))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * num_batch - i)
                    print('\tspeed: {:.6f}s/iter; left time: {:.6f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            valid_loss,reg_valid_loss = self.eval(epoch)
            if 'hpo' in self.args and self.args.hpo == 'optuna':
                # optuna
                self.trial.report(valid_loss.item(), epoch)
            elif 'hpo' in self.args and self.args.hpo == 'nni':
                # nni
                nni.report_intermediate_result(valid_loss.item())
            print("Epoch: {} | cost time: {} | label_valid_loss: {} | reg_valid_loss: {}".format(epoch + 1, time.time() - epoch_time, valid_loss, reg_valid_loss))
            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_checkpoints(epoch)
                num_worse = 0
            else:
                num_worse += 1
            if num_worse > self.args.patience:
                print("Early stop with best valid_loss: {}".format(valid_loss))
                break
            self.stepper.step()
        if 'hpo' in self.args and self.args.hpo == 'nni':
            nni.report_final_result(best_valid_loss.item())
        return best_valid_loss.item()

    def eval(self, epoch):
        valid_data, valid_loader = self._build_dataloader(self.args, self.args.validset_csv_path, 1, shuffle=False, drop_last=False, init_scaler=False, key='valid')
        self.to_eval()
        outputs_list = []
        target_list = []
        with torch.no_grad():
            for i, (samples_batch, targets_batch, datetimes_batch) in enumerate(valid_loader):

                samples_batch = samples_batch.float().to(self.device)
                targets_batch = targets_batch.float().to(self.device)
                # targets_batch = targets_batch.float()
                datetimes_batch = datetimes_batch.float().to(self.device)
                if self.args.timestamp_feature != 'none':
                    outputs = self.predict(samples_batch[:,:self.args.seq_len,:], datetimes_batch[:,:self.args.seq_len,:])
                else:
                    outputs = self.predict(samples_batch[:,:self.args.seq_len,:], None)
                outputs_list.append(outputs)
                target_list.append(targets_batch[:,-self.args.pred_len:,:])
        predict_outputs = torch.cat(outputs_list, dim=0)
        target_outputs = torch.cat(target_list, dim=0)

        eval_result = self.eval_metric(predict_outputs, target_outputs, epoch)
        return eval_result

    def test(self, key):
        self.load_checkpoints(key)
        training_data, training_loader = self._build_dataloader(self.args, self.args.trainset_csv_path, self.args.batch_size, key='train')
        test_data, test_loader = self._build_dataloader(self.args, self.args.testset_csv_path, 1, shuffle=False, drop_last=False, init_scaler=False, key='test')
        self.to_eval()
        outputs_list = []
        target_list = []
        with torch.no_grad():
            for i, (samples_batch, targets_batch, datetimes_batch) in enumerate(test_loader):
                samples_batch = samples_batch.float().to(self.device)
                targets_batch = targets_batch.float().to(self.device)
                datetimes_batch = datetimes_batch.float().to(self.device)
                if self.args.timestamp_feature != 'none':
                    outputs = self.predict(samples_batch[:,:self.args.seq_len,:], datetimes_batch[:,:self.args.seq_len,:])
                else:
                    outputs = self.predict(samples_batch[:,:self.args.seq_len,:], None)
                outputs_list.append(outputs)
                target_list.append(targets_batch[:,-self.args.seq_len:,:])

        target_scaler = test_data.target_scaler
        #unscale = lambda x: x * target_scaler.var_ + target_scaler.mean_
        
        predict_outputs = torch.cat(outputs_list, dim=0)
        predict_outputs = predict_outputs[:,-1:,:]
        # * from scratch
        predict_outputs = target_scaler.inverse_transform(predict_outputs.reshape([len(outputs_list), 1]).cpu())
        # * finetune
        # predict_outputs = target_scaler.inverse_transform(predict_outputs.reshape([len(outputs_list), 57]).cpu())
        predict_outputs = predict_outputs[:,-1:]

        target_outputs = torch.cat(target_list, dim=0)
        target_outputs = target_outputs[:,-1:,:]
        # * from_scratch
        target_outputs = target_scaler.inverse_transform(target_outputs.reshape([len(outputs_list), 1]).cpu())
        # * finetune
        # target_outputs = target_scaler.inverse_transform(target_outputs.reshape([len(outputs_list), 57]).cpu())
        target_outputs = target_outputs[:,-1:]
        
        # predict_outputs = torch.cat(outputs_list, dim=0)
        # target_outputs = torch.cat(target_list, dim=0)

        data = {'predict': predict_outputs.reshape(predict_outputs.shape[0]*predict_outputs.shape[1]), 'groundthuth': target_outputs.reshape(target_outputs.shape[0]*predict_outputs.shape[1])}
        print("data:",data)
        # data = {'predict': predict_outputs.reshape(predict_outputs.shape[0]*predict_outputs.shape[1],predict_outputs.shape[2]), 'groundthuth': target_outputs.reshape(target_outputs.shape[0]*predict_outputs.shape[1],predict_outputs.shape[2])}
        df = pd.DataFrame(data)
        df.to_excel('%s/%s_%s.xlsx' % (self.args.checkpoints,self.model_id, self.args.key))

        # todo: track both label and regression loss in testing stage (Not urgent at the moment)
        if self.args.stage=='train_reg':
            test_result, reg_test_result = self.test_metric(predict_outputs, target_outputs)
            print("test result: {} | regression test result: {}. ".format(test_result.item(),reg_test_result.item()))
        
        test_result = self.test_metric(predict_outputs, target_outputs)
        print("test result: %.3f. " % test_result.item())
        return test_result


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

