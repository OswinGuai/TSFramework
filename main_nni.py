import argparse
import torch
from factory.transformer_forecasting import TransformerForecasting
from factory.alumina_transformer_forecasting import AluminaTransformerForecasting
import random
import numpy as np
import nni
import logging

logger = logging.getLogger('Transformer')

torch.autograd.set_detect_anomaly(True)

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


parser = argparse.ArgumentParser(description='Time Series Forecasting')

parser.add_argument('--use_nni', action="store_true", help='status')
parser.add_argument('--stage', type=str, required=True, default='train', help='status')
parser.add_argument('--model_id', type=str, required=True, help='model id')
parser.add_argument('--log_path', type=str, required=True, help='log')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--key', type=str, help='model id')

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
parser.add_argument('--interval', type=int, default=900, help='')

parser.add_argument('--model_name', type=str, required=True, default='transformer')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--hidden_dim', type=int, default=256, help='num of encoder layers')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
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

def main(params):

    args = argparse.Namespace(**params)
    print('Args in experiment:')
    print(args)
    # set experiments
    factory_list = [TransformerForecasting, AluminaTransformerForecasting]
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
        pred_rmse = forecasting_model.test(args.key)
        print('>>>>>>>end of testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model_id))

    torch.cuda.empty_cache()

try:
    if base_args.use_nni:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        trial_id = nni.get_trial_id()
        logger.debug(tuner_params)
        tuner_params['model_id'] = '%s_%s' % (base_args.model_id, trial_id)
        tuner_params['hpo'] = 'nni'
        print('model_id: %s' % tuner_params['model_id'])
        params = vars(base_args)
        params.update(tuner_params)
        main(params)
    else:
        params = vars(base_args)
        main(params)
except Exception as exception:
    logger.exception(exception)
    raise


