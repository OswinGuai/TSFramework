import pandas as pd
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')

class TimeseriesDataset(Dataset):

    def __init__(self, csv_path, segment_len, granularity, feature_cols, target_cols, datetime_col='Time', timestamp_feature='none'):

        # init
        self.csv_path = csv_path
        self.segment_len = segment_len
        self.granularity = granularity
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.datetime_col = datetime_col
        self.timestamp_feature = timestamp_feature

        self.__read_data__()

    def __read_data__(self):

        data = pd.read_csv(self.csv_path)
        data = data.dropna()
        data = data.reset_index(drop=True)

        data_features = data[self.feature_cols]
        data_target = data[self.target_cols]
        data_datetime = data[self.datetime_col]

        # get temporally length-fixed samples from data_features
        data_len = len(self.data_features)
        gap = self.granularity * self.segment_len
        start_indices = []
        for i in range(0, data_len):
            if i + self.segment_len >= data_len:
                break
            if data_datetime[i] - data_datetime[i + self.segment_len] == gap:
                start_indices.append(i)
            else:
                continue

        data_datetime = pd.to_datetime(data_datetime, unit='ms')
        data_datetime = data_datetime.dt.tz_localize('Asia/Shanghai')
        if self.timestamp_feature == 'day':
            data_datetime = data_datetime.dt.date
        elif self.timestamp_feature == 'hour':
            data_datetime = data_datetime.dt.hour
        elif self.timestamp_feature == 'minute':
            data_datetime = data_datetime.dt.minute
        elif self.timestamp_feature == 'second':
            data_datetime = data_datetime.dt.second

        temporal_samples = []
        temporal_targets = []
        temporal_datetime = []
        for start_index in start_indices:
            end_index = start_index + self.segment_len
            temporal_samples.append(data_features.iloc[start_index:end_index].values)
            temporal_targets.append(data_target.iloc[start_index:end_index].values)
            temporal_datetime.append(data_datetime[start_index:end_index].values)

        self.temporal_samples = temporal_samples
        self.temporal_targets = temporal_targets
        self.temporal_datetime = temporal_datetime

    def __getitem__(self, index):
        samples = self.temporal_samples[index]
        targets = self.temporal_targets[index]
        datetimes = self.temporal_datetime[index]

        # standardize targets
        return samples, targets, datetimes

    def __len__(self):
        return len(self.temporal_samples)

