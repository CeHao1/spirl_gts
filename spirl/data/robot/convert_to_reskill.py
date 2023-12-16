import h5py
import os
import numpy as np
from tqdm import tqdm
from spirl.utils.general_utils import ParamDict, AttrDict



def load_rollout(file_dir):
    F =  h5py.File(file_dir, 'r') 

    data = {}
    key = "traj0"
    for name in F[key].keys():
            if name in ['states', 'actions', 'pad_mask', 'reward', 'done']:
                data[name] = F[key + '/' + name][()].astype(np.float32)
    return data

raw_data_path = "/home/cehao/cehao/maze"
reskill_data_dir = "/home/cehao/cehao/maze_reskill_data.npy"

all_file_list= [] 
for root, dirs, files in os.walk(raw_data_path):
    for file in files:
        if os.path.splitext(file)[1] == '.h5':
            all_file_list.append(os.path.join(root, file))

print("number of files: ", len(all_file_list))

seqs = []
for file in tqdm(all_file_list):
    # 1. load files in the directory
    # 2. parse the data
    # 3. save to the reskill format

    data = load_rollout(file)

    seqs.append(AttrDict(
                    obs=data['states'],
                    actions=data['actions'],
                    ))

        
np_seq = np.array(seqs)
np.save(reskill_data_dir, np_seq)


