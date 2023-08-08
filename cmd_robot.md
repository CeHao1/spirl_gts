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
python3 spirl/rl/train.py --path=spirl/configs/rl/table_cleanup/SAC --seed=0 --gpu=0 \
--prefix=test_s0_02

# convert data to skill
python spirl/data/robot/convert.py 

# train skills
python3 spirl/train.py --path=spirl/configs/skill_prior_learning/table_cleanup/hierarchical --val_data_size=160 \
--gpu=0 --prefix=ol_test_02

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/table_cleanup/hierarchical_cl --val_data_size=160 \
--gpu=0 --prefix=cl_test_01

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/table_cleanup/hierarchical_cd --val_data_size=160 \
--gpu=0 --prefix=cd_test_01

# train agent
python3 spirl/rl/train.py --path=spirl/configs/hrl/table_cleanup/spirl  --gpu=0 \
--seed=0 --prefix=HL_s2_01