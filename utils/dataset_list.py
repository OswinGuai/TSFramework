import os
import csv

import pandas as pd


def get_dataset_info(path):
    dataset_info = {}
    folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
    global_count = 0
    
    for folder in folders:
        folder_path = os.path.join(path, folder)
        csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
        dataset_info[folder] = {}
        
        for csv_file in csv_files:
            csv_file_path = os.path.join(folder_path, csv_file)
            with open(csv_file_path, 'r') as file:
                reader = csv.reader(file)
                rows = sum(1 for row in reader)
                file.seek(0)
                columns = len(next(reader))

            dataset_info[folder][csv_file] = {'rows': rows, 'columns': columns}
            data_count = rows * columns
            global_count += data_count
            print(dataset_info[folder][csv_file], " file_size: ", os.path.getsize(csv_file_path), " data_count: ", data_count)

    print("dataset global size: ", global_count)
    return dataset_info


if __name__ == '__main__':
    # 计算全过程的时间
    import time
    start_time = time.time()
    path = '/data/ts2b'
    dataset_info = get_dataset_info(path)
    print(dataset_info)
    # 请你写一个逻辑将dataset_info保存为json文件
    import json
    with open('tmp.json', 'w') as f:
        print("saving ts2b_info to json file")
        json.dump(dataset_info, f)



    # for folder, datasets in dataset_info.items():
    #     print(f"Folder: {folder}")
    #     for dataset, info in datasets.items():
    #         print(f"Dataset: {dataset}")
    #         print(f"Rows: {info['rows']}")
    #         print(f"Columns: {info['columns']}")
    #         print()
    end_time = time.time()
    print(f"Time: {end_time - start_time}s")
