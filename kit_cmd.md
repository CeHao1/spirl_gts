
export EXP_DIR=./experiments
export DATA_DIR=./data

# train skill


python3 spirl/train.py --path=spirl/configs/skill_prior_learning/kitchen/hierarchical_cl --gpu=0 \
     --val_data_size=160 --prefix=train01

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/kitchen/hierarchical --gpu=0 \
     --val_data_size=160 --prefix=train01


# train agent

python3 spirl/rl/train.py --path=spirl/configs/hrl/kitchen/spirl_cl --seed=0 \
    --prefix=SPIRL_kitchen_seed0_02

