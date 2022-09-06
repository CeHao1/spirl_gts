

## prefix name rules

type_feature_index, sac_nz10_01



export EXP_DIR=./experiments
export DATA_DIR=./data



=====================================================================================
# Train model
python3 spirl/train.py \
--val_data_size=160 --gpu=0 --prefix=xxx_t01 --resume=latest

1. flat model
--path=spirl/configs/skill_prior_learning/gts/flat 

2. open-loop
python3 spirl/train.py \
--val_data_size=160 --gpu=0 --prefix=ol_newdesk_01 \
--path=spirl/configs/skill_prior_learning/gts/hierarchical --resume=latest

3. close-loop
--path=spirl/configs/skill_prior_learning/gts/hierarchical_cl

4. state-conditioned decoder(time-indexed)
python3 spirl/train.py \
--val_data_size=160 --gpu=0 --prefix=cd_newdesk_01  \
--path=spirl/configs/skill_prior_learning/gts/hierarchical_cd --resume=latest


=====================================================================================
# Train RL agent
python3 spirl/rl/train.py \
--gpu=0  --prefix=xxx_01 --resume=latest 

1. SAC (sac_autoalp_01, sac_targetE1_01)
 --path=spirl/configs/rl/gts/SAC 
python3 spirl/rl/train.py --path=spirl/configs/rl/gts/SAC --gpu=0  --prefix=sac_newdesk_01 --resume=latest \
--mode='val' --deterministic_action


python3 spirl/rl/train.py --path=spirl/configs/rl/gts/SAC --prefix=sac_newdesk_01 

2. SAC + BC
--path=spirl/configs/rl/gts/prior_initialized/bc_finetune/ 

3. open-loop spirl
python3 spirl/rl/train.py --gpu=0 --prefix=ol_newobs_01 \
--path=spirl/configs/hrl/gts/spirl \

4. close-loop spirl
--path=spirl/configs/hrl/gts/spirl_cl

5. state-conditioned decoder
--path=spirl/configs/hrl/gts/spirl_cd

6. skill-critic
--path=spirl/configs/hrl/gts/sc
python3 spirl/rl/train.py --path=spirl/configs/hrl/gts/sc --gpu=0  --prefix=sc_noprior_05 \
--resume=latest
--mode='rollout' --save_dir='./sample/hrl/sc_02'



## eval
--mode='val'

## sample rollout
python3 spirl/rl/train.py --path=xx --prefix=xx --gpu=0 --resume='latest' --mode=rollout --deterministic_action=1 --save_dir='./sample/rl/sac/maf' --n_val_samples=50 --counter=0

=====================================================================================
# visualization
--gpu=0 --path=spirl/configs/skill_prior_learning/gts/hierarchical --resume=latest

1. vis model
%run spirl/vis/vis_mdl.py 

2. vis rl
%run spirl/vis/vis_rl.py 

3. vis hrl
%run spirl/vis/vis_hrl.py 



=====================================================================================
## maze
export EXP_DIR=./experiments
export DATA_DIR=/media/cehao/Data/ubuntu_backup/spirl_data

### tarin maze close loop, prior
python3 spirl/train.py --gpu=0 --path=spirl/configs/skill_prior_learning/maze/hierarchical_cl --val_data_size=160 --prefix=cl_maze_01 --resume=latest

### train maze open loop, prior
python3 spirl/train.py --gpu=0 --path=spirl/configs/skill_prior_learning/maze/hierarchical --val_data_size=160 --prefix=ol_maze_test

### train maze cl, spirl
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/spirl_cl/ --prefix=cl_maze_01 --gpu=0 --resume=latest




++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## The structure of agent and policy(in spirl)

                | no prior          |  spirl                            | spirl cl          |   

 LL model       | SkillPriorMdl     |                                   | ClSPiRLMdl        | 
 LL policy      |                   |                                   | ClModelPolicy     | 
 LL agent       | SkillSpaceAgent   |                                   | SACAgent          |

 HL policy      | MLPPolicy         | LearnedPriorAugmentedPIPolicy     |                   |
 HL agent       | SACAgent          | ActionPriorSACAgent               |                   |

 Encoder        |                   | BaseProcessingLSTM                | BaseProcessingLSTM
 Decoder        |                   | RecurrentPredictor                | Predictor
                                      (ForwardLSTMCell, CustomLSTM)

 Joint agent    | FixedIntervalHierarchicalAgent


## The strucure of agent and policy (new algorithm)

 Method         | state-con decoder     | time-indexed decoder  | skill-critic          |
________________|_______________________|_______________________|_______________________|
 LL model       | CDSPiRLMdl            |              TimeIndexCDSPiRLMDL              |
 LL policy      | CDModelPolicy         | TimeIndexedCDMdlPolicy| DecoderRegu_TimeIndexedCDMdlPolicy |
 LL agent       | SACAgent              | SACAgent              | LLActionAgent         |
________________|_______________________|_______________________|_______________________|
 HL policy      |                     LearnedPriorAugmentedPIPolicy                     |
 HL agent       |             ActionPriorSACAgent               | HLSKillAgent          |
________________|_______________________|_______________________|_______________________|
 Encoder        | LSTM  
 Decoder        | Predictor
 Prior          | Predictor

 Joint agent    | JointAgent(FixedIntervalTimeIndexedHierarchicalAgent)

 ## explicit skill-critic structure

High-level:
Function:           at high-level step, choose skills(continuous) z based on the current state
HL_Policy:          z ~ PI_z(s), initialized by the skill prior
HL_Critic:          Qz(s,z,k0)  k0 denotes when k=0
HL_Policy update:   Loss = Qz(s,z,k0) - alp_z*DKL(PI_z||prior)
HL_replay:          (st, zt, st+1)

Low-level:
Function:           at each step, given state, skill and hl step k, choose the action
LL_Policy:          a ~ PI_a(s,z,k), initialized by the decoder
LL_Critic:          Qa(s,z,k,a) 
LL_Policy update:   Loss = Q(s,z,k,a) - alp_a*DKL(PI_a||decoder)
HL_Critic TD Err:   Qz(s,z,k) = Qa(s,z,k,a) - alp_a*DKL(PI_a||decoder)
LL_Critic TD Err:   Qa(s,z,k,a) = r + gam*U, U = (1-beta)*Qz(t+1) + (beta)*Vz(t+1), Vz = Qz(s,z,k) - alp_z*DKL(PI_z||prior)