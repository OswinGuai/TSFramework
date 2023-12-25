"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""
import argparse
import torch

from factory.transformer_forecasting import TransformerForecasting

import random
import numpy as np
import os

import optuna
from optuna.trial import TrialState

import threading
mutex = threading.Lock()

parser = argparse.ArgumentParser(description='Time Series Forecasting')

parser.add_argument('--stage', type=str, required=True, default='train', help='status')
parser.add_argument('--model_id', type=str, required=True, help='model id')
parser.add_argument('--log_path', type=str, required=True, help='log')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--trainset_csv_path', type=str, required=True, help='')
parser.add_argument('--validset_csv_path', type=str, required=True, help='')
parser.add_argument('--testset_csv_path', type=str, required=True, help='')
parser.add_argument('--datetime_col', type=str, help='')
parser.add_argument('--feature_cols', type=str, help='')
parser.add_argument('--target_cols', type=str, help='')
parser.add_argument('--timestamp_feature', type=str, help='')


parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of encoder')
parser.add_argument('--label_len', type=int, default=12, help='start token length of decoder')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--interval', type=int, help='')

parser.add_argument('--model_name', type=str, required=True, default='transformer')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--hidden_dim', type=int, default=256, help='num of encoder layers')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

parser.add_argument('--optimizer', type=str, default='Adam', help='')
parser.add_argument('--fix_seed', type=int, default=-1, help='')
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

base_args = parser.parse_args()
base_args.use_gpu = True if torch.cuda.is_available() else False

used_num_gpu = [0 for _ in range(8)]

def main(params):

    args = argparse.Namespace(**params)
    print('Args in experiment:')
    print(args)
    # set experiments
    factory_list = [TransformerForecasting]
    forecasting_model = None
    for f in factory_list:
        if args.model_name in f.model_choices.keys():
            forecasting_model = f(args)
            break

    if forecasting_model is None:
        assert(False)

    if args.stage == 'train':
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model_id))
        forecasting_model.fit()
        print('>>>>>>>end of training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model_id))
    elif args.stage == 'test':
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.model_id))
        pred_rmse = forecasting_model.eval()
        print('>>>>>>>end of testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model_id))

    torch.cuda.empty_cache()


def objective(trial):
    tuner_params = {
            'lr': trial.suggest_categorical("lr", [0.001,0.0001,0.0003]),
            'd_model': trial.suggest_categorical("d_model", [128,256,512]),
            'hidden_dim': trial.suggest_categorical("hidden_dim", [128,256,512]),
            'e_layers': trial.suggest_int("e_layers", 1, 3, log=True),
            'd_layers': trial.suggest_int("d_layers", 1, 3, log=True),
            }

    with mutex:
        sort_index = [i for i, x in sorted(enumerate(used_num_gpu), key=lambda x: x[1])]
        used_num_gpu[sort_index[0]] += 1
        tuner_params['gpu'] = sort_index[0]

    trial_id = trial.number
    # get parameters form tuner
    tuner_params['model_id'] = '%s_%s' % (base_args.model_id, trial_id)
    tuner_params['hpo'] = 'optuna'
    tuner_params['trial'] = trial
    params = vars(base_args)
    params.update(tuner_params)
    best_valid_loss = main(params)
    used_num_gpu[sort_index[0]] -= 1

    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return best_valid_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", storage="sqlite:///db.sqlite3", study_name=base_args.model_id)
    study.optimize(objective, n_trials=100, timeout=2000, n_jobs=8)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

