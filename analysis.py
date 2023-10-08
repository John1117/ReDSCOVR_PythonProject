import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as tc
from torch import nn

def plot_Kp_correlation(df, input_col_names, mean, std, model, input_window, label_window, offset):

    input_col_names = input_col_names
    if input_col_names is None:
        input_col_names = [name for name in df.columns]

    start_idxs = range(0, len(df) - input_window - offset - label_window + 2)

    labels = []
    preds = []
    for i in start_idxs:
        input_idx = slice(i, i + input_window)
        input = tc.tensor(df[input_col_names][input_idx].to_numpy()).unsqueeze(0)
        pred = model(input, return_series=False).squeeze().detach().numpy() * std + mean

        label_idx = slice(i + input_window + offset - 1, i + input_window + offset - 1 + label_window)
        label = df['Kp'][label_idx].to_numpy().squeeze() * std + mean

        preds.append(pred)
        labels.append(label)

    r = np.corrcoef(preds, labels)[0, 1]
    plt.figure(figsize=(10, 10))
    plt.title(f'Kp correlation = {r:.7f}', fontsize=30)
    plt.plot(preds, labels, 'b.', alpha=0.2)
    plt.plot([0, 9], [0, 9], 'k--')
    plt.xlabel('True Kp', fontsize=30)
    plt.ylabel('Pred Kp', fontsize=30)
    plt.xticks(ticks=range(10), labels=range(10), fontsize=20)
    plt.yticks(ticks=range(10), labels=range(10), fontsize=20)
    plt.show()
    

def plot_Kp_persistence(df, mean, std, hr_shift=1):

    Kps = df['Kp'][:-hr_shift] * std + mean
    Kp_plus1s = df['Kp'][hr_shift:] * std + mean

    r = np.corrcoef(Kps, Kp_plus1s)[0, 1]
    plt.figure(figsize=(10, 10))
    plt.title(f'{hr_shift}hr Kp persistence = {r:.7f}', fontsize=30)
    plt.plot(Kps, Kp_plus1s, 'b.', alpha=0.2)
    plt.plot([0, 9], [0, 9], 'k--')
    plt.xlabel('Kp(t)', fontsize=30)
    plt.ylabel(f'Kp(t+{hr_shift}hr)', fontsize=30)
    plt.xticks(ticks=range(10), labels=range(10), fontsize=20)
    plt.yticks(ticks=range(10), labels=range(10), fontsize=20)
    plt.show()

def grad_analysis(df, input_col_names, mean, std, model, input_window, label_window, offset):
    loss_fn = nn.MSELoss()
    input_col_names = input_col_names
    if input_col_names is None:
        input_col_names = [name for name in df.columns]

    start_idxs = range(0, len(df) - input_window - offset - label_window + 2)

    n_feature = len(df.columns)
    grad_running_sum = np.zeros(n_feature)

    for i in start_idxs:
        input_idx = slice(i, i + input_window)
        input = tc.tensor(np.expand_dims(df[input_col_names][input_idx], 0), requires_grad=True)
        pred = model(input)

        #label_idx = slice(i + input_window + offset - 1, i + input_window + offset - 1 + label_window)
        #label = tc.tensor(np.expand_dims(df['Kp'][label_idx], [0, 1]))
        #loss = loss_fn(pred, label)

        pred.backward()
        grad = input.grad

        grad_running_sum += np.abs(grad.detach().numpy().sum(axis=1).squeeze())

    grad_running_sum /= len(start_idxs)

    n_bar = n_feature//2
    sort_idxs = grad_running_sum[:n_bar].argsort()[::-1]

    bar_width = 0.4
    
    bar_idxs = np.arange(n_bar)
    plt.figure(figsize=(25, 5))
    plt.title('Gradient analysis', fontsize=30)
    plt.bar(bar_idxs, np.abs(grad_running_sum)[:n_bar][sort_idxs], color='b', width=bar_width, label='Value')
    plt.bar(bar_idxs+bar_width, np.abs(grad_running_sum)[n_bar:][sort_idxs], color='c', width=bar_width, label='Check')
    plt.xlabel('Column', fontsize=30)
    plt.ylabel('Absolute gradient', fontsize=30)
    plt.xticks(ticks=bar_idxs+bar_width/2, labels=df.columns[sort_idxs], fontsize=20, rotation=90)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.show()

def nan_analysis(model, inputs, labels, col_names, mean, std):
    loss_fn = nn.MSELoss()
    losses = np.zeros(70, dtype=np.float64)
    for feature_idx in range(70):
        set_nan_inputs = inputs.clone().detach()
        set_nan_inputs[:, :, feature_idx] = 0
        set_nan_inputs[:, :, 70 + feature_idx] = 0

        loss_running_sum = 0
        for i, (input, label) in enumerate(zip(set_nan_inputs, labels)):
            input = input.unsqueeze(0)
            label = label.unsqueeze(0)
            pred = model(input)

            loss = loss_fn(pred, label)

            loss_running_sum += loss

        losses[feature_idx] = loss_running_sum / len(inputs)

    n_bar = 70
    bar_width = 0.4

    sort_idxs = losses.argsort()[::-1]
    
    bar_idxs = np.arange(n_bar)
    plt.figure(figsize=(25, 5))
    plt.title('Set-to-nan analysis', fontsize=30)
    plt.bar(bar_idxs, losses[sort_idxs], color='b', width=bar_width, label='Loss')
    plt.xlabel('Set-to-nan column', fontsize=30)
    plt.ylabel('Loss', fontsize=30)
    plt.xticks(ticks=bar_idxs, labels=col_names[:70][sort_idxs], fontsize=20, rotation=90)
    plt.yticks(fontsize=20)
    plt.show()