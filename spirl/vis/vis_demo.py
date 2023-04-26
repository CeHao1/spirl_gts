import os
import joblib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from spirl.utils.general_utils import AttrDict, listdict2dictlist, batch_listdict2dictlist


# def listdict2dictlist(LD):
#     """ Converts a list of dicts to a dict of lists """
    
#     # Take intersection of keys
#     keys = ['steering', 'throttle', 'brake']
#     return {k: [dic[k] for dic in LD] for k in keys}

def load_data(data_dir):
    file_names = os.listdir(data_dir)
    
    all_list_dict = []
    idx = 0
    for file in tqdm(file_names):
        file_dir = os.path.join(data_dir, file)
        state_one_car = joblib.load(file_dir)
        states_one_car = listdict2dictlist(state_one_car)
        all_list_dict.append(states_one_car)

        idx += 1
        # if idx > 10:
        #     break
        
    state_dict_all = batch_listdict2dictlist(all_list_dict)
    plot_all_dist(state_dict_all)
    
    # plot
    # plot_dist(actions, len(actions))
    
def plot_all_dist(state_dict_all):

    for state in state_dict_all:
        fig = plt.figure(figsize=(10, 8))
        sns.histplot(state_dict_all[state], bins=50)
        plt.title(state, fontsize=16)
        plt.savefig('/home/msc/cehao/github_space/spirl_gts/images/' + state + '.png')


def plot_dist(action, size):
    fs = 15

    fig = plt.figure(figsize=(10, 8))
    sns.histplot(x=action[-size:, 0], y=action[-size:, 1], bins=50, cmap='Reds', cbar=True)
    plt.xlabel('dim_0', fontsize=fs)
    plt.ylabel('dim_1', fontsize=fs)
    plt.title('2D dist of latent variables, size ' + str(size), fontsize=fs)
    plt.show()
    # plt.close(fig)


    fig = plt.figure(figsize=(10, 5))
    fig.tight_layout()
    for idx in range(len(action[0])):
        sns.histplot(action[-size:, idx], bins=50, label='dim_' + str(idx), alpha=0.5)
    plt.legend(loc='upper left', fontsize=14, framealpha=0.5)
    plt.ylabel('Density', fontsize=fs)
    plt.title('distribution of latent variables, size ' + str(size), fontsize=fs)
    plt.grid()
    plt.show()
    # plt.close(fig)
    
if __name__ == "__main__":
    load_data('./sample/raw_data/batch_0')