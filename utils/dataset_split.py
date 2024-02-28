import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import numpy as np


split_judge_len = 3000
split_judge_cols = 100
split_stack = []
n_clusters = 3

if __name__ == '__main__':
    root_path = '/data/ts2b-500M-100'
    
    folders = sorted([f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))])
    for folder in folders:
        folder_path = os.path.join(root_path, folder)
        dataset_files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith('.csv') or f.endswith('.txt') or f.endswith('.npz')])

        for dataset_file in dataset_files:
            dataset_path = os.path.join(folder_path, dataset_file)
            split_stack.append(dataset_path)
            while len(split_stack) > 0:
                dataset_path = split_stack.pop()
                if dataset_path.endswith('.csv'):
                    df_raw = pd.read_csv(dataset_path)
                elif dataset_path.endswith('.txt'):
                    df_raw = []
                    with open(dataset_path, "r", encoding='utf-8') as f:
                        for line in f.readlines():
                            line = line.strip('\n').split(',')
                            data_line = np.stack([float(i) for i in line])
                            df_raw.append(data_line)
                    df_raw = np.stack(df_raw, 0)
                    df_raw = pd.DataFrame(df_raw)

                elif dataset_path.endswith('.npz'):
                    data = np.load(dataset_path, allow_pickle=True)
                    data = data['data'][:, :, 0]
                    df_raw = pd.DataFrame(data)
                else:
                    raise ValueError('Unknown data format: {}'.format(dataset_path))

                # 如果df_raw的的列数小于split_judge_cols，则不进行split
                if len(df_raw.columns) < split_judge_cols:
                    continue

                # 如果df_raw的长度超过split_judge_len，则df_raw只取前split_judge_len行作为判断依据df，否则用全量df_raw作为判断依据df
                df = df_raw.iloc[:split_judge_len, :] if len(df_raw) > split_judge_len else df_raw

                print(f"======splitting {dataset_path}  cols_num: {len(df_raw.columns)}======")
                # read_csv
                timestamp_col = None
                if pd.to_datetime(df.iloc[:, 0], errors='coerce').notna().all():
                    timestamp_col = df_raw.iloc[:, 0]
                    df = df.iloc[:, 1:]
                    df_raw = df_raw.iloc[:, 1:]
                # calculate_similarity
                data = df.values.T
                similarity = cosine_similarity(data)
                # cluster_columns
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(similarity)
                # save_clusters_to_csv
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    columns_for_label = df_raw.columns[labels == label]
                    df_cluster = df_raw[columns_for_label]
                    if timestamp_col is not None:
                        df_cluster.insert(0, timestamp_col.name, timestamp_col)
                    # 去掉dataset_path的后缀
                    dataset_name = f"{dataset_path.split('.')[0]}_{label}.csv"
                    print(f"{dataset_name} cols_num: {len(columns_for_label)}")
                    df_cluster.to_csv(dataset_name, index=False)
                    split_stack.append(os.path.join(folder_path, dataset_name))

                # 删除原文件
                os.remove(dataset_path)
