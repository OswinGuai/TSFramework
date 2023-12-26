from factory.forecasting import GeneralForecasting
from models import *
from data.alumina_dataloader import AluminaDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn


class AluminaTransformerForecasting(GeneralForecasting):
    model_choices = {
            'alumina_transformer': Transformer,
            'alumina_itransformer': iTransformer,
            'alumina_patchtst': PatchTST,
            'alumina_lstm': LSTM,
            }

    def __init__(self, args):
        model_params = {
            'enc_in': len(args.feature_cols.split(',')),
            'dec_in': len(args.feature_cols.split(',')),
            'c_out': len(args.feature_cols.split(',')) + len(args.target_cols.split(',')),
            }
        super().__init__(args, model_params)
        self.criterion = nn.MSELoss()

    def _build_dataloader(self, args, csv_path, batch_size, shuffle=True, num_workers=1, drop_last=True, init_scaler=True, key='train'):
        return self._create_data(args, csv_path, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, init_scaler=init_scaler, pattern=key)

    def _create_data(self, args, csv_path, batch_size, shuffle=True, num_workers=1, drop_last=True, init_scaler=True, pattern='train'):
        dataset = AluminaDataset(
                csv_path=csv_path,
                segment_len=(args.pred_len + args.seq_len),
                # segment_len=30,
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
        rmse = ((eval_outputs - truth_values) ** 2).mean() ** 0.5
        return rmse

    def eval_metric(self, eval_outputs, truth_values, epoch):
        eval_outputs = eval_outputs.cpu()
        truth_values = truth_values.cpu()
        
        print("Truth Value Shape:\n",truth_values.shape)
        print("Eval Output Shape:\n",eval_outputs.shape)
        # if self.pattern == 'pretrain':
        #     mask = torch.ones(truth_values.shape[0])
        # if self.pattern == 'train':
        mask = torch.where(truth_values.reshape(
            [truth_values.shape[0]*truth_values.shape[1]*truth_values.shape[2]]) == -999, 
            torch.zeros(truth_values.shape[0]*truth_values.shape[1]*truth_values.shape[2]), 
            torch.ones(truth_values.shape[0]*truth_values.shape[1]*truth_values.shape[2])
        )
        if torch.sum(mask) == 0:
            raise Exception('testset is mistaken')
        rmse = (torch.sum(mask * ((eval_outputs - truth_values) ** 2).reshape(mask.shape)) / torch.sum(mask)) ** 0.5
        self.writer.add_scalar('eval/rmse', rmse, epoch)
        return rmse

    def _forward(self, samples_end, datetimes_enc,  samples_dec, datetimes_dec):
        model_outputs = self.model(samples_end, datetimes_enc,  samples_dec, datetimes_dec)
        return model_outputs

    def predict(self, samples_batch, datetimes_batch):
        samples_enc = samples_batch
        #datetimes_enc = datetimes_batch
        B, S, V = samples_enc.shape
        samples_dec = torch.cat((samples_enc[:, -self.args.label_len:, :], torch.zeros([B, self.args.pred_len, V]).float().to(self.device)), 1)
        #datetimes_dec = torch.cat((datetimes_enc[:, -self.args.label_len:, :], torch.zeros([B, self.args.pred_len, datetimes_enc.shape[2]]).float().to(self.device)), 1)
        #outputs = self._forward(samples_enc, datetimes_enc,  samples_dec, datetimes_dec)
        outputs = self._forward(samples_enc, None,  samples_dec, None)
        label_outputs = outputs[:,:,-1:]
        return label_outputs

    def batch_loss(self, samples_batch, targets_batch, datetimes_batch, curr_iter):
        samples_enc = samples_batch[:,:self.args.seq_len,:]
        #datetimes_enc = torch.zeros([samples_batch.shape[0], self.args.seq_len, 1]).float().to(self.device)
        samples_dec = torch.cat((samples_enc[:, -self.args.label_len:, :], samples_batch[:, :-self.args.pred_len, :]), 1)
        #datetimes_dec = torch.cat((datetimes_enc[:, -self.args.label_len:, :], torch.zeros([samples_batch.shape[0], self.args.pred_len, 1]).to(self.device).float()), 1)
        outputs = self._forward(samples_enc, None,  samples_dec, None)
        reg_outputs = outputs[:,:,:-1]
        label_outputs = outputs[:,:,-1:]
        reg_targets = samples_batch[:,:,-1:]
        label_targets = targets_batch[:, -self.args.pred_len:, :]
        reg_loss = self.criterion(reg_outputs, reg_targets)
        # print("Label Targets Shape:\n",label_targets.shape)
        mask = torch.where(
            label_targets.reshape([label_targets.shape[0]*label_targets.shape[1]*label_targets.shape[2]]) == -999, 
            torch.zeros(label_targets.shape[0]*label_targets.shape[1]*label_targets.shape[2]).to(self.device), 
            torch.ones(label_targets.shape[0]*label_targets.shape[1]*label_targets.shape[2]).to(self.device)
        )
        if torch.sum(mask) == 0:
            train_loss = reg_loss
            self.writer.add_scalar('loss/label_loss', 0, curr_iter)
            return -999
        else:
            label_loss = (torch.sum(mask * ((label_targets - label_outputs) ** 2).reshape(mask.shape)) / torch.sum(mask)) ** 0.5
            #train_loss = 5 * label_loss + reg_loss
            train_loss = 5 * label_loss + 0 * reg_loss
            self.writer.add_scalar('loss/label_loss', label_loss.item(), curr_iter)
        self.writer.add_scalar('loss/reg_loss', reg_loss.item(), curr_iter)
        self.writer.add_scalar('loss/train_loss', train_loss.item(), curr_iter)
        return train_loss

