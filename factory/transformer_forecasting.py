from factory.forecasting import GeneralForecasting
from models import *

import torch
import torch.nn as nn


class TransformerForecasting(GeneralForecasting):
    model_choices = {
            'transformer': Transformer,
            }

    def __init__(self, args):
        model_params = {
            'enc_in': len(args.feature_cols.split(',')),
            'dec_in': len(args.feature_cols.split(',')),
            'c_out': len(args.target_cols.split(',')),
            }
        super().__init__(args, model_params)
        self.criterion = nn.MSELoss()

    def eval_metric(self, eval_outputs, truth_values, epoch):
        rmse = ((eval_outputs - truth_values) ** 2).mean() ** 0.5
        self.writer.add_scalar('eval/rmse', rmse, epoch)
        return rmse

    def _forward(self, samples_end, datetimes_enc,  samples_dec, datetimes_dec):
        model_outputs = self.model(samples_end, datetimes_enc,  samples_dec, datetimes_dec)
        return model_outputs

    def predict(self, samples_batch, datetimes_batch):
        samples_enc = samples_batch
        datetimes_enc = datetimes_batch
        B, S, V = samples_enc.shape
        samples_dec = torch.cat((samples_enc[:, -self.args.label_len:, :], torch.zeros([B, self.args.pred_len, V]).float().to(self.device)), 1)
        datetimes_dec = torch.cat((datetimes_enc[:, -self.args.label_len:, :], torch.zeros([B, self.args.pred_len, datetimes_enc.shape[2]]).float().to(self.device)), 1)
        outputs = self._forward(samples_enc, datetimes_enc,  samples_dec, datetimes_dec)
        return outputs

    def batch_loss(self, samples_batch, targets_batch, datetimes_batch, curr_iter):
        samples_enc = samples_batch[:,:self.args.seq_len,:]
        datetimes_enc = datetimes_batch[:,:self.args.seq_len]
        samples_dec = torch.cat((samples_enc[:, -self.args.label_len:, :], samples_batch[:, :-self.args.pred_len, :]), 1)
        datetimes_dec = torch.cat((datetimes_enc[:, -self.args.label_len:, :], datetimes_batch[:, :-self.args.pred_len, :]), 1)
        outputs = self._forward(samples_enc, datetimes_enc,  samples_dec, datetimes_dec)
        targets = targets_batch[:, -self.args.pred_len:, :]
        train_loss = self.criterion(outputs, targets)
        self.writer.add_scalar('loss/train_loss', train_loss.item(), curr_iter)
        return train_loss
