


export EXP_DIR=./experiments
export DATA_DIR=./data


# sample data
python spirl/gts_demo_sampler/sample_demo.py \
    --path spirl/configs/data_collect/gts/time_trial/c2 \
    --ip_address '192.168.1.126' \
    --prefix 'batch_8'



# learn skill

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/gts_corner2/hierarchical_cd --val_data_size=160 \
--gpu=0 --prefix=HitWall_hasNorm

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/gts_corner2/flat --val_data_size=160 \
--gpu=0 --prefix=flat



## vis demo states
python spirl/vis/vis_mdl.py --path=spirl/configs/skill_prior_learning/gts_corner2/hierarchical_cd 

# train RL

### SAC single
python3 spirl/rl/train.py --path=spirl/configs/rl/gts_corner2/SAC_single --seed=0 --gpu=0 \
--prefix=sac_new_t01

### SAC multi
python3 spirl/rl/train.py --path=spirl/configs/rl/gts_corner2/SAC_new --seed=1 --gpu=0 \
--prefix=dense_s1_01

### BC multi
python3 spirl/rl/train.py --path=spirl/configs/rl/gts_corner2/prior_initialized/bc_multi --seed=4 --gpu=0 \
--prefix=test_s5

### HRL single
python3 spirl/rl/train.py --path=spirl/configs/hrl/gts_corner2/sh --seed=0 --gpu=0 \
--prefix=rollout_test2 --mode='rollout' --save_dir='./save_rollout'

### HRL multi

python3 spirl/rl/train.py --path=spirl/configs/hrl/gts_corner2/sh_multi --seed=0 --gpu=0 \
--prefix=HL_s2_01

python3 spirl/rl/train.py --path=spirl/configs/hrl/gts_corner2/sh_multi --seed=0 --gpu=0 \
--prefix=HLVar_Norm_s0_13

python3 spirl/rl/train.py --path=spirl/configs/hrl/gts_corner2/sh_multi  --gpu=0 \
--seed=2 --prefix=HYB_td80_s2_21 \
--resume='latest' --strict_weight_loading=0 --resume_load_replay_buffer=0 



