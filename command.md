

export EXP_DIR=./experiments 
export DATA_DIR=./data

export DISPLAY=:0

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze/hierarchical --val_data_size=160 \
--gpu=0 --prefix=ol_maze_init01

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze/hierarchical_cl --val_data_size=160 \
--gpu=0 --prefix=cl_maze_init01

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/spirl_cl --seed=0 --gpu=0 \
--prefix=cl_paper199_seed0

mpi4py                   3.1.4