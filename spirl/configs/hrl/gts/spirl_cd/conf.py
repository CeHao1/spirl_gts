
from spirl.configs.hrl.gts.spirl_cl.conf import *
from spirl.models.cond_dec_spirl_mdl import CDSPiRLMdl, TimeIndexCDSPiRLMDL
from spirl.rl.policies.cd_model_policy import CDModelPolicy

# create LL conditioned decoder policy
ll_policy_params = AttrDict(
    policy_model=CDSPiRLMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "skill_prior_learning/gts/hierarchical_cd"),
)
ll_policy_params.update(ll_model_params)
ll_policy_params.update(ll_model_params)

ll_agent_config = AttrDict(
    policy=CDModelPolicy,
    policy_params=ll_policy_params,
    critic=SplitObsMLPCritic,
    critic_params=ll_critic_params
)