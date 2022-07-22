# Accelerating Reinforcement Learning with Learned Skill Priors



export EXP_DIR=./experiments
export DATA_DIR=./data

=====================================================================================
# for state conditioned 
## train model
python3 spirl/train.py --gpu=0 --path=spirl/configs/skill_prior_learning/gts/hierarchical_cd --val_data_size=160 --prefix=cd_gts_t01 --resume=latest
## train agent
python3 spirl/rl/train.py --gpu=0 --path=spirl/configs/hrl/gts/spirl_cd/ --prefix=cd_gts_t01  --resume=latest

## for skill critic
python3 spirl/rl/train.py --gpu=0 --path=spirl/configs/hrl/gts/sk/ --prefix=csk_gts_01  --resume=latest

=====================================================================================
## test close loop dim
export EXP_DIR=./experiments
export DATA_DIR=/media/cehao/Data/ubuntu_backup/spirl_data

### tarin maze close loop, prior
python3 spirl/train.py --gpu=0 --path=spirl/configs/skill_prior_learning/maze/hierarchical_cl --val_data_size=160 --prefix=cl_maze_01 --resume=latest

### train maze open loop, prior
python3 spirl/train.py --gpu=0 --path=spirl/configs/skill_prior_learning/maze/hierarchical --val_data_size=160 --prefix=ol_maze_test

### train maze cl, spirl
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/spirl_cl/ --prefix=cl_maze_01 --gpu=0 --resume=latest



=====================================================================================
## Train skill priors
### Train no close loop prior
python3 spirl/train.py --gpu=0 --path=spirl/configs/skill_prior_learning/gts/hierarchical --val_data_size=160 --prefix=ol_gts_bs128_wqhat1e-3 --resume=latest


### train gts close loop, prior
python3 spirl/train.py --gpu=0 --path=spirl/configs/skill_prior_learning/gts/hierarchical_cl --val_data_size=160 --prefix=cd_gts_ori --resume=latest


### viz
%run spirl/viz/viz_mdl.py --gpu=0 --path=spirl/configs/skill_prior_learning/gts/hierarchical --resume=latest


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
python3 spirl/rl/train.py --path=spirl/configs/hrl/gts/spirl/ --prefix=ol_hier_sample_test --gpu=0 --resume=latest

### sample_rollout
python3 spirl/rl/train.py --path=spirl/configs/hrl/gts/spirl --prefix=sp_oldbatch --gpu=0  --mode=rollout --save_dir='./sample/hrl/spirl' --n_val_samples=1 --resume=latest

### viz
%run spirl/viz/viz_hrl.py --gpu=0 --path=spirl/configs/hrl/gts/spirl --prefix=sp_03 --resume=latest


### train close loop spirl
python3 spirl/rl/train.py --path=spirl/configs/hrl/gts/spirl_cl/ --prefix=cd_gts_llvar02 --gpu=0 --resume=latest

=====================================================================================
## No prior
### Train
python3 spirl/rl/train.py --path=spirl/configs/hrl/gts/no_prior/ --seed=0 --prefix=np_new_prior2 --gpu=0 --resume=latest






++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## The structure of agent and policy

                | no prior          |  spirl                            | spirl cl          |   sc + ll

 LL model       | SkillPriorMdl     |                                   | ClSPiRLMdl        | CDSPiRLMdl TimeIndexCDSPiRLMDL
 LL policy      |                   |                                   | ClModelPolicy     | 
 LL agent       | SkillSpaceAgent   |                                   | SACAgent          |

 HL policy      | MLPPolicy         | LearnedPriorAugmentedPIPolicy     |                   |
 HL agent       | SACAgent          | ActionPriorSACAgent               |                   |

 Encoder        |                   | BaseProcessingLSTM                | BaseProcessingLSTM
 Decoder        |                   | RecurrentPredictor                | Predictor
                                      (ForwardLSTMCell, CustomLSTM)

 Joint agent    | FixedIntervalHierarchicalAgent



ForwardLSTMCell(CustomLSTMCell) 可以人为的初始化每个层的 hidden variable
CustomLSTM 自己写的lstm，是base
BaseProcessingLSTM 这个东西是不支持 输出 hidden variable 的

state-conditioned:sc, low-level fine tuning: ll
需要增加的部分：

ActionPriorSACAgent, 用来掌管critic的，用的KLD 代替entropy。需要修改 value, critic_loss的计算方法，分别为HL和LL。
其中LL 的V 就是HL的Q， HL的V也用来更新LL的Q。
因此（1）要区分HL，LL的 SACagent，（2）要把他们的critic互相存储， （3）LL要能够利用obs split之后恢复k，判断beta

LearnedPriorAugmentedPIPolicy 这是HL 的policy，貌似没变可以用。
但是在LL的 ClModelPolicy 需要结合， LearnedPriorAugmentedPIPolicy 的初始化和reg。
LL 的agent 也用 ActionPriorSACAgent 拓展出来的结果。

然后joint agent 也要修改成为对应的。

1. 创建新的 joint agent, HL agent, LL agent, HL policy, LL policy
2. joint agent 需要将 HL agent, LL agent 内的critic相互同步
3. 写出正确的skill-critic 的 policy loss, Q loss
4. 考虑一下update 时候的lr 和 batch_size