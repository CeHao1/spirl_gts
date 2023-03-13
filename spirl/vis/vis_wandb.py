

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

# import seaborn as sns

def read_wandb_csv(path, xlim=None, boolean=False):
    data = pd.read_csv(path)

    keys = list(data.keys())
    step = data['Step'].to_numpy()
    items = keys[1::3] # remove the MIN, MAX
    data = data[items].to_numpy()

    data = np.nan_to_num(data)

    if boolean:
        data[data > 0] = 1

    if xlim:
        data = data[step < xlim, :]
        step = step[step < xlim]

    return step, data


def calculate_distribution(data, smooth_sigma=2):
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)

    mean_smooth = gaussian_filter1d(mean, sigma=smooth_sigma)
    upper_err = gaussian_filter1d(mean + (std/2), sigma=smooth_sigma)
    lower_err = gaussian_filter1d(mean - (std/2), sigma=smooth_sigma)
    return mean_smooth, upper_err, lower_err


def plot_smooth_success_rate(path_list, label_list=[], xlim=1e6):

    assert len(path_list) == len(label_list)

    color_list = ['b', 'r', 'g']

    plt.figure(figsize=(7,5))

    for idx in range(len(path_list)):
        step, data = read_wandb_csv(path_list[idx], xlim, boolean=True)
        step = step / 1e6
        data = data * 100

        mean, up_std, dn_std = calculate_distribution(data, smooth_sigma=10)
        
        plt.plot(step, mean, color=color_list[idx], label=label_list[idx])
        plt.fill_between(step, up_std, dn_std, color=color_list[idx], alpha=0.1)


    fs = 15
    plt.grid()
    plt.legend(fontsize=fs)
    plt.xlabel('Environment Steps (1M)', fontsize=fs)
    plt.ylabel('Success Rate [%]', fontsize=fs)
    plt.savefig("reward")
    plt.show()


if __name__ == "__main__":
    path_list = [
    "wandb_export_2023-03-13T10_45_22.623+08_00.csv",
    "wandb_export_2023-03-13T11_44_50.544+08_00.csv",
    "wandb_export_2023-03-13T12_11_18.545+08_00.csv"]

    label_list = ['cl', 'ol', 'cl_noT']

    plot_smooth_success_rate(path_list, label_list)


