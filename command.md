

export EXP_DIR=./experiments
export DATA_DIR=./data

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/cehao/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


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

python3 spirl/rl/train.py --path=spirl/configs/rl/maze/SAC_m2 --seed=1 --gpu=0 \
--prefix=test_s2

python3 spirl/rl/train.py --path=spirl/configs/rl/maze/prior_initialized/bc_m1 --seed=0 --gpu=0 \
--prefix=test_01


python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/sh  --gpu=0 \
--seed=1 --prefix=m1_s1_05

# in maze0
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL  --gpu=0 \
--seed=0 --prefix=m0lv2_HLsLLQ_s0_02

# in maze1

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m1  --gpu=0 \
--seed=2 --prefix=HL_s2_01

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m1  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=abl2_HYB_LLH_s0_01 --seed=0


# in maze2
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--seed=1 --prefix=HL_s1_01

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--seed=0 --prefix=HYB_Noinit_s0_01

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=pureLL_s2_01 --seed=2


## 
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=LL_td10_Var-3_s0_01 --seed=0

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=LL_td20_Var-3_s0_01 --seed=0

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=LL_td50_Var-3_s0_01 --seed=0

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=LL_td80_Var-3_s0_01 --seed=0

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=LL_td80_Var-1_s0_01 --seed=0

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=LL_td80_Var-5_s0_01 --seed=0

rests 3x6 = 18

td 10, 20, 50, 80
Var -5, -3, -1




sh scripts/m2LLtd10Var-3.sh


sh scripts/m2LLtd20Var-3.sh


sh scripts/m2LLtd50Var-3.sh


sh scripts/m2LLtd80Var-3.sh


sh scripts/m2LLtd80Var-1.sh


sh scripts/m2LLtd80Var-5.sh


