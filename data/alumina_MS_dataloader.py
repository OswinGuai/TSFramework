import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import warnings
from sklearn.preprocessing import StandardScaler
import torch
import pdb

warnings.filterwarnings('ignore')

class AluminaMSDataset(Dataset):

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

        if self.init_scaler:
            self.feature_scaler.fit(data_features)
            self.target_scaler.fit(data_target)
        self.data_features = self.feature_scaler.transform(data_features)
        self.data_target = self.target_scaler.transform(data_target)

        # get temporally length-fixed samples from data_features
        data_len = len(data_features)
 
        start_indices = [i for i in range(int(data_len)-self.segment_len + 1)]
        self.start_indices = start_indices



    def __getitem__(self, index):
        posi = int(self.start_indices[index])
        targets=[]
        samples = self.data_features[posi : posi + self.seq_len]

        targets = self.data_target[posi + self.seq_len : posi + self.seq_len + self.pred_len]
        
        return samples, np.array(targets, dtype=float), 0
        

    def __len__(self):
        return len(self.start_indices)


