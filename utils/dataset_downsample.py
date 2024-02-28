import pandas as pd


def downsample_csv_file(file_path, ratio=0.1):
    # 读取csv文件
    df = pd.read_csv(file_path)
    interval = int(1 / ratio)
    # 每十行取一行
    df_downsample = df.iloc[::interval]

    # 创建新的文件名
    new_file_path = file_path.rsplit('.', 1)[0] + '_downsample.csv'

    # 保存到新的csv文件
    df_downsample.to_csv(new_file_path, index=False)


# 使用函数
downsample_csv_file('/data/ts2b-medium/Wind_Power/ts0.csv', ratio=0.1)