import argparse
import torch
from factory.transformer_forecasting import TransformerForecasting
from factory.alumina_transformer_forecasting import AluminaTransformerForecasting
from factory.alumina_MS_forecasting import AluminaTransformerMSForecasting
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
parser.add_argument('--task_name', type=str, default='Forecast', help='type of running task.')
parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')


parser.add_argument('--trainset_csv_path', type=str, required=True, help='')
parser.add_argument('--validset_csv_path', type=str, required=True, help='')
parser.add_argument('--testset_csv_path', type=str, required=True, help='')
parser.add_argument('--datetime_col', type=str, help='')
parser.add_argument('--feature_cols', type=str, help='')
parser.add_argument('--target_cols', type=str, help='')
parser.add_argument('--timestamp_feature', type=str, help='')
parser.add_argument('--random_features',type=bool, default=False, help='Turn this to True if choosing random features in dataset')
parser.add_argument('--random_features_num',type=int, default=10, help='The number of random selected features, only available when: random_features == True')

parser.add_argument('--seq_len', type=int, default=30, help='input sequence length of encoder')
parser.add_argument('--label_len', type=int, default=12, help='start token length of decoder')
parser.add_argument('--pred_len', type=int, default=30, help='prediction sequence length')
parser.add_argument('--interval', type=int, default=900, help='')

parser.add_argument('--model_name', type=str, required=True, default='transformer')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
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
parser.add_argument('--label_loss_rate', type=float, default=0.5, help='the label loss ratio of train loss in train_reg stage')
parser.add_argument('--reg_loss_rate', type=float, default=0.5, help='the regression loss ratio of train loss in train_reg stage')

parser.add_argument('--use_norm', type=int, default=1, help='instance norm')
parser.add_argument('--patch_len', type=int, default=16, help='input sequence length')
parser.add_argument('--target_num', type=int, default=1, help='target num')

base_args = parser.parse_args()
base_args.use_gpu = True if torch.cuda.is_available() else False

def main(params):

    args = argparse.Namespace(**params)

    if(args.random_features==True):
            allfeatures = args.feature_cols.split(',')
            random_selected_features = random.sample(allfeatures,args.random_features_num)
            print("Random Selected Features: ",random_selected_features)
            args.feature_cols = ','.join(random_selected_features) + "," +args.target_cols
            print("Feature Columns in Main: ", args.feature_cols)

    print('Args in experiment:')
    print(args)
    # set experiments
    if args.features == 'M' or args.features == 'MS':
        factory_list = [TransformerForecasting, AluminaTransformerMSForecasting]
        print('Performing MS Forecasting (TimeXer)')
        
    else:
        factory_list = [TransformerForecasting, AluminaTransformerForecasting]
        print('Performing Original Forecasting')
    forecasting_model = None
    for f in factory_list:
        if args.model_name in f.model_choices.keys():
            forecasting_model = f(args)
            break

    if forecasting_model is None:
        assert(False)

    if args.stage == 'pretrain':
        print('>>>>>>>start pretraining : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model_id))
        forecasting_model.prefit()
        print('>>>>>>>end of pretraining : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model_id))
    elif args.stage == 'train':
        print('>>>>>>>start finetuning : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model_id))
        forecasting_model.fit(args.key)
        print('>>>>>>>end of finetuning : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model_id))
    elif args.stage == 'train_only':
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model_id))
        forecasting_model.fit_only()
        print('>>>>>>>end of training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model_id))
    elif args.stage == 'train_reg':
        print('>>>>>>>start training with regression: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model_id))
        forecasting_model.fit_reg()
        print('>>>>>>>end of training with regression: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model_id))
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
        logger.info(tuner_params)
        tuner_params['model_id'] = '%s/%s' % (base_args.model_id, trial_id)
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


