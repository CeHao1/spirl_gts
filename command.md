

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


python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/sh  --gpu=0 \
--seed=1 --prefix=m1_s1_05

# in maze0
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL  --gpu=0 \
--seed=0 --prefix=m0lv2_HLsLLQ_s0_02

# in maze1

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m1  --gpu=0 \
--seed=0 --prefix=HL_s0

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m1  --gpu=0 \
--seed=0 --prefix=LL_td80_real \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0



# in maze2
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--seed=0 --prefix=HYB_td10

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--seed=0 --prefix=HYB_LLV-1_02 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0