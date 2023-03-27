

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
--seed=0 --prefix=sh_lv1_nosquash_s0_01

# train noT skill

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze_bar/hierarchical --val_data_size=160 \
--gpu=0 --prefix=ol_mazeB_init01

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze_bar/hierarchical_cl --val_data_size=160 \
--gpu=0 --prefix=cl_mazeB_init01

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze_bar/hierarchical_cd --val_data_size=160 \
--gpu=0 --prefix=cd_mazeB_init01

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze_h/hierarchical_cd --val_data_size=160 \
--gpu=0 --prefix=cd_mazeh_init01

# train noT RL

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze_bar/spirl --seed=2 --gpu=0 \
--prefix=ol_mB_vis_s2_01

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze_bar/spirl_cl --seed=5 --gpu=0 \
--prefix=cl_mB_base_s5

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze_bar/sc --seed=0 --gpu=0 \
--prefix=sc_mB_s0

<!-- no T -->
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze_noT/sh --seed=0 --gpu=0 \
--prefix=mT_lv1_nosquash_s0_01

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze_bar/sh --seed=0 --gpu=0 \
--prefix=mB_lv1_nosquash_s0_01


python3 spirl/rl/train.py --path=spirl/configs/hrl/maze_h/sh --seed=0 --gpu=0 \
--prefix=mh_lv1_s0_01