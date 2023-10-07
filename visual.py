import numpy as np
import matplotlib.pyplot as plt
import torch as tc
from torch import nn


def plot_demo(
        df, 
        train_mean,
        train_std,
        input_col_names=None, 
        label_col_names=['Kp'], 
        plot_col_name='Kp',
        t_res='1min',
        input_window=1, 
        label_window=1, 
        offset=0, 
        max_demos=3, 
        model=None
    ):

    data_len = len(df)
    max_demos = min(max_demos, data_len)

    demo_idxs = np.random.choice(data_len, size=max_demos, replace=False)

    input_col_names = input_col_names
    if input_col_names is None:
        input_col_names = [name for name in df.columns]

    label_col_names = label_col_names
    if label_col_names is None:
        label_col_names = [name for name in df.columns]

    plot_col_name = plot_col_name
    if plot_col_name is None:
        plot_col_name = label_col_names[0]

    mean = train_mean[plot_col_name]
    std = train_std[plot_col_name]

    if model is not None:
        inputs = []
        for i in demo_idxs:
            input = df[input_col_names][i:i+input_window].to_numpy()
            inputs.append(input)
        inputs = tc.tensor(np.array(inputs))
        preds = model(inputs).detach().numpy() * std + mean

    input_idxs = np.arange(-input_window + 1, 0 + 1)
    label_idxs = np.arange(offset, offset + label_window)
    total_idxs = np.arange(-input_window+1, offset+label_window)

    series = df[plot_col_name] * std + mean

    for j, i in enumerate(demo_idxs):
        plt.figure(figsize=(20, 7.5))
        plt.plot(input_idxs, series[i:i+input_window], 'k.', ms=20, label='Input')
        if plot_col_name in label_col_names:
            plt.plot(label_idxs, series[i+input_window+offset:i+input_window+offset+label_window], 'ko', mfc='w', ms=20, label='Label')
            if model is not None:
                plt.plot(label_idxs, preds[j, :, :], 'b.', ms=20, label='Pred')
        plt.xticks(ticks=total_idxs, labels=total_idxs, fontsize=20)
        plt.yticks(ticks=range(10), labels=range(10), fontsize=20)
        plt.xlabel(f'Time shift (x{t_res})', fontsize=30)
        plt.ylabel(plot_col_name, fontsize=30)
        
        plt.legend(fontsize=20) 
        plt.grid()
        plt.show()


"""import numpy as np
import matplotlib.pyplot as plt
import torch as tc"""


def plot_demo(
        df, 
        train_mean,
        train_std,
        input_col_names=None, 
        label_col_names=['Kp'], 
        plot_col_name='Kp',
        t_res='1min',
        input_window=1, 
        label_window=1, 
        offset=0, 
        max_demos=3, 
        model=None
    ):

    data_len = len(df)
    max_demos = min(max_demos, data_len)

    demo_idxs = np.random.choice(data_len, size=max_demos, replace=False)

    input_col_names = input_col_names
    if input_col_names is None:
        input_col_names = [name for name in df.columns]

    label_col_names = label_col_names
    if label_col_names is None:
        label_col_names = [name for name in df.columns]

    plot_col_name = plot_col_name
    if plot_col_name is None:
        plot_col_name = label_col_names[0]

    mean = train_mean[plot_col_name]
    std = train_std[plot_col_name]

    if model is not None:
        inputs = []
        for i in demo_idxs:
            input = df[input_col_names][i:i+input_window].to_numpy()
            inputs.append(input)
        inputs = tc.tensor(np.array(inputs))
        preds = model(inputs).detach().numpy() * std + mean

    input_idxs = np.arange(-input_window + 1, 0 + 1)
    label_idxs = np.arange(offset, offset + label_window)
    total_idxs = np.arange(-input_window+1, offset+label_window)

    series = df[plot_col_name] * std + mean

    for j, i in enumerate(demo_idxs):
        plt.figure(figsize=(20, 7.5))
        plt.plot(input_idxs, series[i:i+input_window], 'k.', ms=20, label='Input')
        if plot_col_name in label_col_names:
            plt.plot(label_idxs, series[i+input_window+offset:i+input_window+offset+label_window], 'ko', mfc='w', ms=20, label='Label')
            if model is not None:
                plt.plot(label_idxs, preds[j, :, :], 'b.', ms=20, label='Pred')
        plt.xticks(ticks=total_idxs, labels=total_idxs, fontsize=20)
        plt.yticks(ticks=range(10), labels=range(10), fontsize=20)
        plt.xlabel(f'Time shift (x{t_res})', fontsize=30)
        plt.ylabel(plot_col_name, fontsize=30)
        
        plt.legend(fontsize=20) 
        plt.grid()
        plt.show()


def plot_series_pred(df, input_col_names, mean, std, model, start_idx, end_idx, input_window, label_window, offset):
    loss_fn = nn.MSELoss()
    input_idxs = slice(start_idx, end_idx - offset - label_window + 1)
    inputs = tc.tensor(df[input_col_names][input_idxs].to_numpy()).unsqueeze(0)

    pred_idxs = slice(start_idx + offset, end_idx)
    labels = tc.tensor(df['Kp'][pred_idxs].to_numpy()) #.unsqueeze(0)
    preds = model(inputs, return_series=True).squeeze()
    loss = loss_fn(preds, labels)

    pred_ts = df.index[pred_idxs]
    preds = preds.detach().numpy() * std + mean
    plt.plot(pred_ts, preds, 'b-', label='Series pred')
    return np.sqrt(loss.squeeze().detach().numpy())

def plot_batch_pred(df, input_col_names, mean, std, model, start_idx, end_idx, input_window, label_window, offset):
    loss_fn = nn.MSELoss()
    pred_start_idxs = range(start_idx, end_idx - input_window - offset - label_window + 2)
    pred_ts = []
    preds = []
    loss_running_sum = 0
    for i in pred_start_idxs:
        pred_idx = slice(i + input_window + offset - 1, i + input_window + offset - 1 + label_window)
        pred_t = df.index[pred_idx]
        pred_ts.append(pred_t)

        input = tc.tensor(df[input_col_names][i : i + input_window].to_numpy()).unsqueeze(0)
        label = tc.tensor(df['Kp'][pred_idx].to_numpy()).squeeze()
        pred = model(input, return_series=False).squeeze()

        loss = loss_fn(pred, label)
        loss_running_sum += loss
        pred_pt = pred.detach().numpy() * std + mean
        preds.append(pred_pt)
    plt.plot(pred_ts, preds, 'g-', label='Batch pred')
    return np.sqrt(np.sum(loss_running_sum.squeeze().detach().numpy())/len(pred_start_idxs))

def plot_series(
        df,
        train_mean,
        train_std,
        input_col_names=None, 
        label_col_names=['Kp'], 
        plot_col_name='Kp', 
        start=None, 
        end=None,
        input_window=1, 
        label_window=1, 
        offset=0, 
        model=None,
        pred_type='both',
    ):
    
    data_len = len(df)

    input_col_names = input_col_names
    if input_col_names is None:
        input_col_names = [name for name in df.columns]

    label_col_names = label_col_names
    if label_col_names is None:
        label_col_names = [name for name in df.columns]

    plot_col_name = plot_col_name
    if plot_col_name is None:
        plot_col_name = label_col_names[0]

    start_idx = 0
    if isinstance(start, str):
        start_idx = df.index.get_indexer([start], method='nearest')[0]
    elif isinstance(start, int):
        if start < 0:
            start_idx = start + data_len
        else:
            start_idx = start
    
    end_idx = data_len
    if isinstance(end, str):
        end_idx = df.index.get_indexer([end], method='nearest')[0]
    elif isinstance(end, int):
        if end < 0:
            end_idx = end + data_len
        else:
            end_idx = end

    mean = train_mean[plot_col_name]
    std = train_std[plot_col_name]
    series = df[plot_col_name] * std + mean
    
    data_idxs = slice(start_idx, end_idx)
    data_ts = df.index[data_idxs]
    data_pts = series[data_idxs]

    plt.figure(figsize=(20, 10))

    plt.plot(data_ts, data_pts, 'k-', mfc='w', label='Data')

    title = ''
    if model is not None and plot_col_name in label_col_names:
        if pred_type == 'series':
            series_rmse = plot_series_pred(df, input_col_names, mean, std, model, start_idx, end_idx, input_window, label_window, offset)
            title = f'Series RMSE = {series_rmse:.7f}'
        elif pred_type == 'batch':
            batch_rmse = plot_batch_pred(df, input_col_names, mean, std, model, start_idx, end_idx, input_window, label_window, offset)
            title = f'Batch RMSE = {batch_rmse:.7f}'
        elif pred_type == 'both':
            series_rmse = plot_series_pred(df, input_col_names, mean, std, model, start_idx, end_idx, input_window, label_window, offset)
            batch_rmse = plot_batch_pred(df, input_col_names, mean, std, model, start_idx, end_idx, input_window, label_window, offset)
            title = f'RMSE (series, batch) = ({series_rmse:.7f}, {batch_rmse:.7f})'

    plt.title(title, fontsize=20)
    plt.legend(fontsize=20)
    plt.xlabel('Time', fontsize=40)
    plt.ylabel(plot_col_name, fontsize=40)
    plt.xticks(fontsize=20, rotation=-60)
    plt.yticks(ticks=range(10), labels=range(10), fontsize=20)
    plt.ylim(-0.2, data_pts.max()+1)
    plt.grid()
    plt.show()