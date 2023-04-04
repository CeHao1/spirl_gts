

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
--prefix=cl_paper199_seed0_01

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/spirl --seed=0 --gpu=0 \
--prefix=ol_nosquash_s0_01

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/spirl_cl --seed=0 --gpu=0 \
--prefix=cl_vis_s0

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/sc  --gpu=0 \
--seed=1 --prefix=sc_nosquash_s1_01

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/sh  --gpu=0 \
--seed=1 --prefix=m1_s1_05

# train noT skill

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze_bar/hierarchical --val_data_size=160 \
--gpu=0 --prefix=ol_mazeB_init01

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze_bar/hierarchical_cl --val_data_size=160 \
--gpu=0 --prefix=cl_mazeB_init01

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze_bar/hierarchical_cd --val_data_size=160 \
--gpu=0 --prefix=cd_mazeB_init01


# test new maze1 env
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/sh_m1  --gpu=0 \
--seed=1 --prefix=mo_mv6_s1_01

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze_h/sh_m1  --gpu=0 \
--seed=0 --prefix=mh_mv6_s0_01

# train LL as well
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL  --gpu=0 \
--seed=2 --prefix=log-3_s2_02

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze_h/shLL  --gpu=0 \
--seed=0 --prefix=mh_s0_01

# in maze1
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze_h/shLL_m1  --gpu=0 \
--seed=0 --prefix=mh_m1_s0_01