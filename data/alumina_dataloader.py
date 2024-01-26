import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import warnings
from sklearn.preprocessing import StandardScaler
import torch
import pdb

warnings.filterwarnings('ignore')

class AluminaDataset(Dataset):

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    def __init__(self, csv_path, segment_len,seq_len,pred_len, feature_cols, target_cols, datetime_col='Time', interval='none', timestamp_feature='none', init_scaler=True, pattern='train'):

        # init
        self.csv_path = csv_path
        self.segment_len = segment_len
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_cols = feature_cols.split(',')
        self.pattern = pattern
        self.target_cols = target_cols.split(',')
        self.datetime_col = datetime_col
        self.interval = interval
        self.timestamp_feature = timestamp_feature
        self.init_scaler = init_scaler
        

        self.__read_data__()

    def __read_data__(self):

        data = pd.read_csv(self.csv_path,encoding="GB2312")
        data = data.reset_index(drop=True)

        data_features = data[self.feature_cols]
        data_target = data[self.target_cols]
        # self.valid_label = np.arange(1, data.shape[0]/30 + 1) * 30 - 1
        self.valid_label = np.arange(1, int(data.shape[0]/120) + 1) * 120 - 1


        if self.init_scaler:
            self.feature_scaler.fit(data_features)
            self.target_scaler.fit(data_target.iloc[self.valid_label])
        self.data_features = self.feature_scaler.transform(data_features)
        self.data_target = self.target_scaler.transform(data_target)

        # get temporally length-fixed samples from data_features
        data_len = len(data_features)
        if self.pattern == 'pretrain':
            start_indices = [i for i in range(int(data_len)-self.segment_len + 1)] # 6,2  0,1 1,2 2,3 3,4 4,5
            self.start_indices = start_indices

        elif self.pattern == 'train' or self.pattern == 'train_only' or self.pattern == 'train_reg':
            # start_indices = [range(i - 60 + 1, i - 30 + 1) for i in self.valid_label]
            start_indices = [range(i - self.segment_len + 1, i - self.pred_len + 1) for i in self.valid_label]
            start_indices = start_indices[:-29]

            self.start_indices = np.concatenate(start_indices)
        else: # self.pattern == 'test'
            self.start_indices = self.valid_label - self.segment_len + 1


    def __getitem__(self, index):
        posi = int(self.start_indices[index])
        targets=[]
        samples = self.data_features[posi : posi + self.seq_len]
        # samples = np.concatenate((samples,np.ones((samples.shape[0],1))*-999),axis=1)
        targets = self.data_target[posi + self.seq_len : posi + self.seq_len + self.pred_len]
        if self.pattern == 'train' or self.pattern == 'train_only' or self.pattern == 'train_reg':
            for i in range(len(targets)):
                if posi + self.seq_len + i not in self.valid_label:
                    targets[i,-1] = -999
            # standardize targets
        return samples, np.array(targets, dtype=float), 0
        

    def __len__(self):
        return len(self.start_indices)


