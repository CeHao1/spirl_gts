
export EXP_DIR=./experiments
export DATA_DIR=./data

# train skill

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/kitchen/hierarchical --gpu=0 \
     --val_data_size=160 --prefix=train01

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/kitchen/hierarchical_cl --gpu=0 \
     --val_data_size=160 --prefix=train01

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/kitchen/hierarchical_cd --gpu=0 \
     --val_data_size=160 --prefix=train01


# train agent

python3 spirl/rl/train.py --path=spirl/configs/hrl/kitchen/spirl --seed=0 \
    --prefix=ol_self_seed0_01

python3 spirl/rl/train.py --path=spirl/configs/hrl/kitchen/spirl_cl --seed=0 \
    --prefix=cl_self_seed0_02

python3 spirl/rl/train.py --path=spirl/configs/hrl/kitchen/sh --seed=0 \
    --prefix=sh_HYB2_s0_01 \
--resume='latest' --strict_weight_loading=0 --resume_load_replay_buffer=0 