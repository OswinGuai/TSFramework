from factory.forecasting import GeneralForecasting
from models import *
from models import TimeXer
from data.alumina_MS_dataloader import AluminaMSDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import time
import nni
import pandas as pd


class AluminaTransformerMSForecasting(GeneralForecasting):
    model_choices = {
            'alumina_transformer': Transformer,
            'alumina_transformer_new': Transformer_new,
            'alumina_itransformer': iTransformer,
            'alumina_patchtst': PatchTST,
            'alumina_lstm': LSTM,
            'alumina_timexer' : TimeXer,
            }

    def __init__(self, args):
        model_params = {
            'enc_in': len(args.feature_cols.split(',')),
            'dec_in': len(args.feature_cols.split(',')),
            'c_out': len(args.target_cols.split(',')),
            }
 

        super().__init__(args, model_params)
        self.criterion = nn.MSELoss()

    def _build_dataloader(self, args, csv_path, batch_size, shuffle=True, num_workers=1, drop_last=True, init_scaler=True, key='train'):
        return self._create_data(args, csv_path, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, init_scaler=init_scaler, pattern=key)

    def _create_data(self, args, csv_path, batch_size, shuffle=True, num_workers=1, drop_last=True, init_scaler=True, pattern='train'):
        dataset = AluminaMSDataset(
                csv_path=csv_path,
                segment_len=(args.pred_len + args.seq_len),
                seq_len = args.seq_len,
                pred_len = args.pred_len,
                feature_cols=args.feature_cols,
                target_cols=args.target_cols,
                datetime_col=args.datetime_col,
                interval=args.interval,
                timestamp_feature=args.timestamp_feature,
                init_scaler=init_scaler,
                pattern=pattern
            )

        dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last)
        return dataset, dataloader

    def test_metric(self, eval_outputs, truth_values):
        if self.args.stage=='pretrain':
            if 'itransformer' in self.args.stage or 'patchtst' in self.args.model_id: # itransformer selected or patchtst selected
                eval_outputs = eval_outputs
                truth_values = truth_values
            else: # transformer selected
                eval_outputs = eval_outputs[:,:-1]
                truth_values = truth_values[:,:-1]
        elif self.args.stage=='train_reg':
            label_eval_outputs = eval_outputs[:,-1:]
            label_truth_values = truth_values[:,-1:]
            label_rmse = ((label_eval_outputs - label_truth_values) ** 2).mean() ** 0.5

            reg_eval_outputs = eval_outputs[:,:-1]
            reg_truth_outputs = truth_values[:,:-1]
            reg_rmse = ((reg_eval_outputs - reg_truth_outputs) ** 2).mean() ** 0.5

            return label_rmse, reg_rmse

        else:# self.args.stage == 'train' or 'train_only' or 'test'
            if 'itransformer' in self.args.stage or 'patchtst' in self.args.model_id: # itransformer selected or patchtst selected
                eval_outputs = eval_outputs
                truth_values = truth_values
            else: # transformer selected
                eval_outputs = eval_outputs[:,-1:]
                truth_values = truth_values[:,-1:]
        rmse = ((eval_outputs - truth_values) ** 2).mean() ** 0.5
        return rmse

    def eval_metric(self, eval_outputs, truth_values, epoch):
        eval_outputs = eval_outputs.cpu()
        truth_values = truth_values.cpu()
        if self.args.stage=='pretrain':
            if 'itransformer' in self.args.stage or 'patchtst' in self.args.model_id: # itransformer selected or patchtst selected
                eval_outputs = eval_outputs
                truth_values = truth_values
            else: # transformer selected
                eval_outputs = eval_outputs[:,:,:-1]
                truth_values = truth_values[:,:,:-1]
        
        elif self.args.stage=='train_reg':
            label_eval_outputs = eval_outputs[:,:,-1:]
            label_truth_values = truth_values[:,:,-1:]

            reg_eval_outputs = eval_outputs[:,:,:-1]
            reg_truth_values = truth_values[:,:,:-1]

            label_rmse = ((label_eval_outputs - label_truth_values) ** 2).mean() ** 0.5

            reg_rmse = self.criterion(reg_eval_outputs,reg_truth_values)

            return label_rmse, reg_rmse

        else:# self.args.stage == 'train' or 'train_only' or 'test'
            if 'itransformer' in self.args.stage or 'patchtst' in self.args.model_id: # itransformer selected or patchtst selected
                eval_outputs = eval_outputs
                truth_values = truth_values
            else: # transformer selected
                # import pdb
                # pdb.set_trace()
                eval_outputs = eval_outputs[:,:,-1:]
                truth_values = truth_values[:,:,-1:]
        
        rmse = ((eval_outputs - truth_values) ** 2).mean() ** 0.5
        self.writer.add_scalar('eval/rmse', rmse, epoch)
        return rmse

    def _forward(self, samples_end, datetimes_enc,  samples_dec, datetimes_dec):
        model_outputs = self.model(samples_end, datetimes_enc,  samples_dec, datetimes_dec)
        return model_outputs

    def predict(self, samples_batch, datetimes_batch):
        samples_enc = samples_batch[:,:self.args.seq_len,:]
        #datetimes_enc = datetimes_batch
        B, S, V = samples_enc.shape
        samples_dec = torch.cat((samples_enc[:, -self.args.label_len:, :], torch.zeros([B, self.args.pred_len, V]).float().to(self.device)), 1)

        #datetimes_dec = torch.cat((datetimes_enc[:, -self.args.label_len:, :], torch.zeros([B, self.args.pred_len, datetimes_enc.shape[2]]).float().to(self.device)), 1)
        #outputs = self._forward(samples_enc, datetimes_enc,  samples_dec, datetimes_dec)
        outputs = self._forward(samples_enc, None,  samples_dec, None)
        # print("predict output shape:",outputs.shape)
        label_outputs = outputs[:,:,:]
        return label_outputs

    def batch_loss(self, samples_batch, targets_batch, datetimes_batch, curr_iter):
        
        samples_enc = samples_batch[:,:self.args.seq_len,:]
        #datetimes_enc = torch.zeros([samples_batch.shape[0], self.args.seq_len, 1]).float().to(self.device)
        # samples_dec = torch.cat((samples_enc[:, -self.args.label_len:, :], samples_batch[:, :-self.args.pred_len, :]), 1)
        B, S, V = samples_enc.shape
        samples_dec = torch.cat((samples_enc[:, -self.args.label_len:, :], torch.zeros([B, self.args.pred_len, V]).float().to(self.device)), 1)

        #datetimes_dec = torch.cat((datetimes_enc[:, -self.args.label_len:, :], torch.zeros([samples_batch.shape[0], self.args.pred_len, 1]).to(self.device).float()), 1)
        outputs = self._forward(samples_enc, None,  samples_dec, None)
       
        # import pdb
        # pdb.set_trace()

        if self.args.stage=='pretrain':
            if 'itransformer' in self.args.model_id or 'patchtst' in self.args.model_id: # itransformer selected or patchtst selected
                reg_outputs = outputs
                reg_targets = targets_batch
            else: # transformer selected
                reg_outputs = outputs[:,:,:-1]
                reg_targets = targets_batch[:,:,:-1]
            # import pdb
            # pdb.set_trace()
            reg_loss = self.criterion(reg_outputs, reg_targets)
            self.writer.add_scalar('loss/reg_loss', reg_loss.item(), curr_iter)
            train_loss = reg_loss
            # label_targets = targets_batch[:, -self.args.pred_len:, :-1]

        elif self.args.stage=='train_reg':
            '''
            calculate both regression loss and label loss in training process
            '''
            reg_outputs = outputs[:,:,:-1]
            reg_targets = targets_batch[:,:,:-1]
            label_outputs = outputs[:,:,-1:]
            label_targets = targets_batch[:, -self.args.pred_len:, -1:]

            reg_loss = self.criterion(reg_outputs, reg_targets)
            self.writer.add_scalar('loss/reg_loss', reg_loss.item(), curr_iter)

            label_loss = self.criterion(label_outputs,label_targets)
            self.writer.add_scalar('loss/label_loss', label_loss.item(), curr_iter)
            train_loss = self.args.label_loss_rate * label_loss + self.args.reg_loss_rate * reg_loss
            # print("train_loss: ",train_loss)
            # print("label_loss: ",label_loss)
            # print("reg_loss: ",reg_loss)
            return train_loss, label_loss, reg_loss

        else: # self.args.stage == 'train' or 'train_only' or 'test'
            if 'itransformer' in self.args.model_id or 'patchtst' in self.args.model_id: # itransformer selected or patchtst selected
                label_outputs = outputs
                label_targets = targets_batch[:,-self.args.pred_len:,:]
            else: #transformer selected
                label_outputs = outputs[:,:,-1:]
                label_targets = targets_batch[:, -self.args.pred_len:, -1:]

            label_loss = self.criterion(label_outputs,label_targets)
            #train_loss = 5 * label_loss + reg_loss
            train_loss = label_loss 
            self.writer.add_scalar('loss/label_loss', label_loss.item(), curr_iter)
        # label_outputs = outputs[:,:,-self.args.pred_len:]
        # reg_targets = samples_batch[:,:,-self.args.pred_len:]
        # label_targets = targets_batch[:, -self.args.pred_len:, :]
            
        self.writer.add_scalar('loss/train_loss', train_loss.item(), curr_iter)
        return train_loss
    
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
                
                self.backward(train_loss)

                if (i + 1) % int(num_batch/5) == 0:
                    
                    print("\titers: {0}, epoch: {1} | train_loss : {2:.4f}".format(i + 1, epoch + 1, train_loss.item()))
                    
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
                
                self.backward(train_loss)

                if (i + 1) % int(num_batch/5) == 0:
                    
                    print("\titers: {0}, epoch: {1} | train_loss : {2:.4f}".format(i + 1, epoch + 1, train_loss.item()))
                    
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
                
                self.backward(train_loss)

                if (i + 1) % int(num_batch/5) == 0:
                    
                    print("\titers: {0}, epoch: {1} | train_loss : {2:.4f}".format(i + 1, epoch + 1, train_loss.item()))
                    
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
                
                self.backward(train_loss)

                if (i + 1) % int(num_batch/5) == 0:
                    
                    print("\titers: {0}, epoch: {1} | train_loss : {2:.4f} | label_loss : {3:.4f} | reg_loss : {4:.4f}".format(i + 1, epoch + 1, train_loss.item(),label_loss.item(),reg_loss.item()))
                   
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
