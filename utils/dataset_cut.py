import pandas as pd


def cut_csv_file(file_path, ratio=0.1):
    # 读取csv文件
    df = pd.read_csv(file_path)

    # 计算需要截取的行数
    rows_to_cut = int(len(df) * ratio)

    # 截取前n行
    df_cut = df.head(rows_to_cut)

    # 创建新的文件名
    new_file_path = file_path.rsplit('.', 1)[0] + '_cut.csv'

    # 保存到新的csv文件
    df_cut.to_csv(new_file_path, index=False)


# 使用函数
cut_csv_file('/data/ts2b-medium/Solar_Power/ts0.csv', ratio=0.1)