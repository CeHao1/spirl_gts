
# Skill-Critic for GTS

This is the code for implementing skill-critic in GTS environment.
For basically installation, please refer to the spirl_readme.md.

# Implementation
To implement skill-critic, we need to first sample demonstration from built-in AI and train skill prior.  
Then use skill-critic to fine-tune the learned skills in downstream RL.

## Environment setup
The experimental environment is to run Audi TTCup on Tokyo Expressway Central Outer Loop. Details can be found in the skill-critic manuscript. Please also initialize GTS (course and car) before running experiments.

## Demo sampling
To sample demonstration, we can use the following command. We should change the ip_address to the local PlayStation. We also enable parallel sampling by changing the prefix name.

```python spirl/gts_demo_sampler/sample_demo.py \
    --path spirl/configs/data_collect/gts/time_trial/c2 \
    --ip_address '192.168.1.xxx' \
    --prefix 'batch_0'
```

Please copy the sampled demo to the data directory. Details are in spirl_readme.md.

## Learning priors
We train flat and hierarchical priors using the following commands.

``` 
python3 spirl/train.py --path=spirl/configs/skill_prior_learning/gts_corner2/flat --val_data_size=160 --gpu=0 --prefix=flat_priors 

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/gts_corner2/hierarchical_cd --val_data_size=160 --gpu=0 --prefix=HRL_priors 
```

After training, we need to copy the learned weights to the expeirment directory.

## Training skill-critic
Our training supports to sample rollouts from more than one PlayStation. Please change the ip_address accordingly in the ```spirl/configs/hrl/gts_corner2/sh_multi```. Please also initialize every GTS before experiments. (A quick way is to set ```do_init = True``` in the conf.py)  

We run skill-critic by 
```python3 spirl/rl/train.py --path=spirl/configs/hrl/gts_corner2/sh_multi --seed=0 --gpu=0 --prefix=skill_critic ```

In config.py, we only train high-level policy if ```agent_config.initial_train_stage = skill_critic_stages.HL_TRAIN```, otherwise when ```agent_config.initial_train_stage = skill_critic_stages.HYBRID```, both high- and low-level polices are trained.

## SAC methods
We can further test SAC and BC+SAC by

```
python3 spirl/rl/train.py --path=spirl/configs/rl/gts_corner2/SAC_new --seed=0 --gpu=0 --prefix=sac  
python3 spirl/rl/train.py --path=spirl/configs/rl/gts_corner2/prior_initialized/bc_multi --seed=0 --gpu=0 --prefix=bc 
```

Please also change the ip_address in their config.py to every PlayStation.
