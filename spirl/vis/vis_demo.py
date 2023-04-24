import os
import joblib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def listdict2dictlist(LD):
    """ Converts a list of dicts to a dict of lists """
    
    # Take intersection of keys
    keys = ['steering', 'throttle', 'brake']
    return {k: [dic[k] for dic in LD] for k in keys}

def load_data(data_dir):
    file_names = os.listdir(data_dir)
    
    action_list = []
    for file in tqdm(file_names):
        file_dir = os.path.join(data_dir, file)
        state_one_car = joblib.load(file_dir)
        states_one_car = listdict2dictlist(state_one_car)
        
        steer2range = np.pi / 6
        action = []
        action.append(np.array(states_one_car['steering']) / steer2range)
        action.append(np.array(states_one_car['throttle']) - np.array(states_one_car['brake']))
        action = np.array(action).T
        action_list += action.tolist()  
    
    actions = np.array(action_list)
    
    # plot
    plot_dist(actions, len(actions))
    
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