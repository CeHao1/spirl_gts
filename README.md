# Accelerating Reinforcement Learning with Learned Skill Priors



export EXP_DIR=./experiments
export DATA_DIR=./data

=====================================================================================
## Train skill priors
### Train no close loop prior
python3 spirl/train.py --gpu=0 --path=spirl/configs/skill_prior_learning/gts/hierarchical --val_data_size=160 --prefix=cus_01 --resume=latest

### viz
%run spirl/viz/viz_mdl.py --gpu=0 --path=spirl/configs/skill_prior_learning/gts/hierarchical --resume=latest

### train customized prior


=====================================================================================
## Train SAC
### Train
python3 spirl/rl/train.py --path=spirl/configs/rl/gts/SAC --prefix=maf6_2 --gpu=0 --resume='latest'

### Eval
python3 spirl/rl/train.py --path=spirl/configs/rl/gts/SAC --prefix=s_new_nn_back --gpu=0  --mode='val' --resume='latest'

### Viz
%run spirl/viz/viz_mdl.py --gpu=0 --path=spirl/configs/rl/gts/SAC --prefix=s_changetable --gpu=0  --resume='latest'


### Sample rollout
python3 spirl/rl/train.py --path=spirl/configs/rl/gts/SAC --prefix=maf6_2 --gpu=0 --resume='latest' --mode=rollout --deterministic_action=1 --save_dir='./sample/rl/sac/maf' --n_val_samples=50 --counter=0

=====================================================================================
## Train spirl agent
### Train agent
python3 spirl/rl/train.py --path=spirl/configs/hrl/gts/spirl/ --prefix=pr_test02 --gpu=0 --resume=latest

### sample_rollout
python3 spirl/rl/train.py --path=spirl/configs/hrl/gts/spirl --prefix=sp_oldbatch --gpu=0  --mode=rollout --save_dir='./sample/hrl/spirl' --n_val_samples=1 --resume=latest

### viz
%run spirl/viz/viz_hrl.py --gpu=0 --path=spirl/configs/hrl/gts/spirl --prefix=sp_03 --resume=latest

=====================================================================================
## No prior
### Train
python3 spirl/rl/train.py --path=spirl/configs/hrl/gts/no_prior/ --seed=0 --prefix=np_new_prior2 --gpu=0 --resume=latest














