import gym
import reskill.rl.envs
env = gym.make("FetchCleanUp-v0")
env.reset()
print('123')
# gym                     0.23.1
# gym==0.12.1


export EXP_DIR=./experiments
export DATA_DIR=./data

# train SAC
python3 spirl/rl/train.py --path=spirl/configs/rl/table_clearnup/SAC --seed=0 --gpu=0 \
--prefix=test_s0_02

# convert data to skill

