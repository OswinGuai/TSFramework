import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class AluminaDataset(Dataset):

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    def __init__(self, csv_path, segment_len, feature_cols, target_cols, datetime_col='Time', interval='none', timestamp_feature='none', init_scaler=True, pattern='train'):

        # init
        self.csv_path = csv_path
        self.segment_len = segment_len
        self.feature_cols = feature_cols.split(',')
        self.pattern = pattern
        # if self.pattern=='pretrain':
        #     self.target_cols = self.feature_cols.split(',')
        # else:
        #     self.target_cols = target_cols.split(',')
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
        self.valid_label = np.arange(1, data.shape[0]/120 + 1) * 120 - 1


        if self.init_scaler:
            self.feature_scaler.fit(data_features)
            self.target_scaler.fit(data_target.iloc[self.valid_label])
        self.data_features = self.feature_scaler.transform(data_features)
        self.data_target = self.target_scaler.transform(data_target)

        # get temporally length-fixed samples from data_features
        data_len = len(data_features)
        if self.pattern == 'pretrain':
            start_indices = [i for i in range(int(data_len)-60)]
            self.start_indices = start_indices

        elif self.pattern == 'train':
            # start_indices = [i * 30 + np.array(list(range(30 - self.segment_len + 1))) for i in range(int(data_len/30))]
            # start_indices = [i * 120 + 60 for i in range(int(data_len/120))]
            start_indices = [i * 120 + np.array(list(range(120 - self.segment_len + 1))) for i in range(int(data_len/120))]
            self.start_indices = np.concatenate(start_indices)
            # self.start_indices = start_indices
        else:
            self.start_indices = self.valid_label - self.segment_len + 1


    def __getitem__(self, index):
        posi = int(self.start_indices[index])
        samples = self.data_features[posi:posi + self.segment_len]
        targets=[]
        if self.pattern == 'pretrain':
            targets = self.data_features[posi+self.segment_len:posi+self.segment_len*2]
            return samples, np.array(targets, dtype=float), 0
        
        # if self.pattern == 'train':
        else:
            targets = self.data_target[posi: posi + self.segment_len]
            for i in range(len(targets)):
                if posi + i not in self.valid_label:
                    targets[i] = -999
            # standardize targets
            return samples, np.array(targets, dtype=float), 0
        

    def __len__(self):
        return len(self.start_indices)


