
# commands for maze env
export EXP_DIR=./experiments
export DATA_DIR=./data


# train sac
python3 spirl/rl/train.py --path=spirl/configs/rl/maze/SAC --seed=0 --prefix=SAC_maze_test_01


# train prior and use it to train RL

## close-loop
python3 spirl/train.py  --gpu=0 --path=spirl/configs/skill_prior_learning/maze/hierarchical_cl --val_data_size=160 --prefix=maze_cl_01
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/spirl_cl --seed=0 --prefix=maze_cl_01 


## open loop
python3 spirl/train.py  --gpu=0 --path=spirl/configs/skill_prior_learning/maze/hierarchical --val_data_size=160 --prefix=maze_ol_01
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/spirl --seed=0 --prefix=maze_ol_01 


## skill-critic
