import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch as tc
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


def resample_t_res(df: pd.DataFrame, t_res: pd.Timedelta('1min')):
    rsp_df = df.resample(rule=t_res).mean()
    return rsp_df


def split_dataframe(df: pd.DataFrame, split_ratio=0.75):
    data_len = len(df)
    
    if isinstance(split_ratio, float):
        split_ratio = np.clip(split_ratio, 0, 1)
        split_idx = int(data_len * split_ratio)
        train_df = df[:split_idx]
        test_df = df[split_idx:]
        return train_df, test_df
    
    elif isinstance(split_ratio, list):
        split_ratio = np.array(split_ratio) / np.sum(split_ratio)

        if len(split_ratio) == 2:
            split_idx = int(data_len * split_ratio[0])
            train_df = df[:split_idx]
            test_df = df[split_idx:]
            return train_df, test_df
        
        elif len(split_ratio) == 3:
            split_idx1 = int(data_len * split_ratio[0])
            split_idx2 = -int(data_len * split_ratio[2])
            train_df = df[:split_idx1]
            valid_df = df[split_idx1:split_idx2]
            test_df = df[split_idx2:]
            return train_df, valid_df, test_df


def standardize_dataframe(train_mean: pd.Series, train_std: pd.Series, train_df: pd.DataFrame, test_df: pd.DataFrame, valid_df=None):
    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    if valid_df is None:
        return train_df, test_df
    else:
        valid_df = (valid_df - train_mean) / train_std
        return train_df, test_df, valid_df
    
def fill_nan(with_nan_dfs, fill_nan_val=0):
    dfs = []
    if isinstance(fill_nan_val, (int, float)):
        for with_na_df in with_nan_dfs:
            df = with_na_df.fillna(fill_nan_val)
            dfs.append(df)

    elif isinstance(fill_nan_val, str):
        if fill_nan_val == 'mean':
            fill_nan_val = with_na_df.mean()
        elif fill_nan_val == 'median':
            fill_nan_val = with_na_df.median()
        for with_na_df in with_nan_dfs:
            df = with_na_df.fillna(fill_nan_val)
            dfs.append(df)
    return dfs


def preprocess_dataframe(df, t_res: pd.Timedelta('1min'), split_ratio=0.75, fill_nan_val=0):
    rsp_df = resample_t_res(df, t_res)
    train_sdf, test_sdf = split_dataframe(rsp_df, split_ratio)

    train_mean = train_sdf.mean()
    train_std = train_sdf.std()
    train_zdf, test_zdf = standardize_dataframe(train_mean, train_std, train_sdf, test_sdf)

    train_df, test_df = fill_nan([train_zdf, test_zdf], fill_nan_val)
    return train_df, test_df, train_mean, train_std


def make_tensordata(train_df, test_df, input_col_names=None, label_col_names=['Kp'], input_window=1, label_window=1, offset=1):
    input_col_names = input_col_names
    if input_col_names is None:
        input_col_names = [name for name in train_df.columns]

    label_col_names = label_col_names
    if label_col_names is None:
        label_col_names = [name for name in train_df.columns]

    train_data_len = len(train_df)
    test_data_len = len(test_df)

    train_inputs = []
    train_labels = []
    for i in range(train_data_len - input_window - label_window - offset):
        input = train_df[input_col_names][i : i + input_window].to_numpy()
        label = train_df[label_col_names][i + input_window + offset - 1 : i + input_window + offset - 1 + label_window].to_numpy()
        train_inputs.append(input)
        train_labels.append(label)
    train_inputs = np.array(train_inputs)
    train_labels = np.array(train_labels)

    test_inputs = []
    test_labels = []
    for i in range(test_data_len - input_window - label_window - offset):
        input = test_df[input_col_names][i : i + input_window].to_numpy()
        label  = test_df[label_col_names][i + input_window + offset - 1 : i + input_window + offset - 1 + label_window].to_numpy()
        test_inputs.append(input)
        test_labels.append(label)
    test_inputs = np.array(test_inputs)
    test_labels = np.array(test_labels)

    return tc.tensor(train_inputs), tc.tensor(train_labels), tc.tensor(test_inputs), tc.tensor(test_labels)