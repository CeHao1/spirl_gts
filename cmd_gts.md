


export EXP_DIR=./experiments
export DATA_DIR=./data

# sample data
python spirl/gts_demo_sampler/sample_demo.py \
    --path spirl/configs/data_collect/gts/time_trial/c2 \
    --ip_address '192.168.1.126' \
    --prefix 'batch_8'



# learn skill

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/gts_corner2/hierarchical_cd --val_data_size=160 \
--gpu=0 --prefix=cedesk_01


# train RL

### SAC single
python3 spirl/rl/train.py --path=spirl/configs/rl/gts_corner2/SAC_single --seed=0 --gpu=0 \
--prefix=sac_new_t01

### SAC multi
python3 spirl/rl/train.py --path=spirl/configs/rl/gts_corner2/SAC_new --seed=0 --gpu=0 \
--prefix=sac_new_t01

### HRL single
python3 spirl/rl/train.py --path=spirl/configs/hrl/gts_corner2/sh --seed=0 --gpu=0 \
--prefix=sac_new_t01

### HRL multi
python3 spirl/rl/train.py --path=spirl/configs/hrl/gts_corner2/sh_multi --seed=0 --gpu=0 \
--prefix=HLsLLQ_s0_31

python3 spirl/rl/train.py --path=spirl/configs/hrl/gts_corner2/sh_multi  --gpu=0 \
--seed=0 --prefix=HYB_35 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0