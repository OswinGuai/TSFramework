import argparse
import torch
import os

from factory.general_forecasting import GeneralForecasting

torch.autograd.set_detect_anomaly(True)
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


parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of encoder')
parser.add_argument('--label_len', type=int, default=12, help='start token length of decoder')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--granularity', type=int, help='')

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

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() else False

print('Args in experiment:')
print(args)

path = "checkpoints/outputs"
isExists=os.path.exists(path)
if not isExists:
    os.makedirs(path)

# set experiments
factory_list = [GeneralForecasting]
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
elif args.stage == 'test':
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.model_id))
    pred_rmse = forecasting_model.eval()

torch.cuda.empty_cache()

