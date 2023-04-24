

export EXP_DIR=./experiments
export DATA_DIR=./data

export DISPLAY=:1

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze/hierarchical --val_data_size=160 \
--gpu=0 --prefix=ol_maze_init01


# train skill
python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze/flat --val_data_size=160 \
--gpu=0 --prefix=flat_maze_01



python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze/hierarchical_cl --val_data_size=160 \
--gpu=0 --prefix=cl_maze_init01

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/maze/hierarchical_cd --val_data_size=160 \
--gpu=0 --prefix=cd_maze_init01


# train RL

python3 spirl/rl/train.py --path=spirl/configs/rl/maze/SAC_m2 --seed=0 --gpu=0 \
--prefix=test_01

python3 spirl/rl/train.py --path=spirl/configs/rl/maze/prior_initialized/bc_m2 --seed=0 --gpu=0 \
--prefix=test_01


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




# test new maze1 env
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/sh_m1  --gpu=0 \
--seed=0 --prefix=mo_md_s0_01

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze_bar/sh_m1  --gpu=0 \
--seed=0 --prefix=mb_md_s0_01

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze_h/sh_m1  --gpu=0 \
--seed=0 --prefix=mh_md_s0_01

# train LL as well
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL  --gpu=0 \
--seed=2 --prefix=log-3_s2_02

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze_h/shLL  --gpu=0 \
--seed=0 --prefix=mh_s0_01

# in maze1

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m1  --gpu=0 \
--seed=0 --prefix=mo_topmid_HLsLLQ

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m1  --gpu=0 \
--seed=4 --prefix=mo_topmid_LLP_s4 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0


# in maze0
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL  --gpu=0 \
--seed=0 --prefix=m0lv2_HLsLLQ_s0_02

# in maze2
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m3  --gpu=0 \
--seed=0 --prefix=HYB_td10

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m3  --gpu=0 \
--seed=1 --prefix=HYB_02_next_02 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0