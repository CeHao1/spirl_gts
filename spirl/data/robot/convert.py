
import numpy as np
from spirl.utils.general_utils import ParamDict, AttrDict
from spirl.gts_demo_sampler.file.file_operation import *

from tqdm import tqdm

fname = '/home/msc/cehao/github_space/reskill/dataset/fetch_block_40000/demos.npy'
rollout_dir = '/home/msc/cehao/github_space/spirl_gts/data/table_cleanup/batch_0'
seqs = np.load(fname, allow_pickle=True)


for idx in tqdm(range(len(seqs))):
    seq = seqs[idx]
    action = np.array(seq.actions)
    observation = np.array(seq.obs)
    image = np.zeros((len(action), 1, 1, 3))
    done = [False for _ in image]
    done[-1] = True

    done = np.array(done)
    episode = AttrDict( observation=observation, action=action , image = image, done=done)
    save_rollout(str(idx), episode, rollout_dir)