

export EXP_DIR=./experiments 
export DATA_DIR=./data

export DISPLAY=:1

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze/hierarchical --val_data_size=160 \
--gpu=0 --prefix=ol_maze_init01


# train skill
python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze/hierarchical_cl --val_data_size=160 \
--gpu=0 --prefix=cl_maze_init01

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze/hierarchical_cd --val_data_size=160 \
--gpu=0 --prefix=cd_maze_init01


# train RL
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/spirl_cl --seed=1 --gpu=0 \
--prefix=cl_paper199_seed1

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/spirl --seed=3 --gpu=0 \
--prefix=ol_iter199_seed3

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/spirl_cl --seed=0 --gpu=0 \
--prefix=cl_iter199_seed0

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/sc --seed=0 --gpu=0 \
--prefix=sc_iter199_seed0

# train noT skill

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze_noT/hierarchical --val_data_size=160 \
--gpu=0 --prefix=ol_mazeT_init01

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze_noT/hierarchical_cl --val_data_size=160 \
--gpu=0 --prefix=cl_mazeT_init01

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze_noT/hierarchical_cd --val_data_size=160 \
--gpu=0 --prefix=cd_mazeT_init01

# train noT RL

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze_noT/spirl --seed=1 --gpu=0 \
--prefix=ol_mT_iter199_seed1

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze_noT/spirl_cl --seed=1 --gpu=0 \
--prefix=cl_mT_iter199_seed1