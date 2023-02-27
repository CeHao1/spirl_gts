
export EXP_DIR=./experiments
export DATA_DIR=./data

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/spirl_cl --seed=0 --prefix=cl_199iter_01 --gpu=0

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/spirl --seed=3 --prefix=ol_iter199_seed3 --gpu=0


python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze/hierarchical --val_data_size=160 --gpu=0 --prefix=ol_maze_init01

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze/hierarchical_cl --val_data_size=160 --gpu=0 --prefix=cl_maze_init01