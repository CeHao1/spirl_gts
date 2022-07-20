
from spirl.configs.hrl.gts.spirl.conf import *
from spirl.models.cond_dec_spirl_mdl import CDSPiRLMdl, TimeIndexCDSPiRLMDL
from spirl.rl.policies.cd_model_policy import CDModelPolicy, TimeIndexedCDMdlPolicy
from spirl.rl.components.critic import SplitObsMLPCritic

from spirl.rl.components.agent import FixedIntervalTimeIndexedHierarchicalAgent

configuration.agent = FixedIntervalTimeIndexedHierarchicalAgent

ll_model_params.cond_decode = True
# create LL conditioned decoder policy
ll_policy_params = AttrDict(
    # policy_model=CDSPiRLMdl,
    policy_model = TimeIndexCDSPiRLMDL,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "skill_prior_learning/gts/hierarchical_cd"),

    manual_log_sigma=[1e-10, 1e-12],
)
ll_policy_params.update(ll_model_params)

ll_critic_params = AttrDict(
    action_dim=data_spec.state_dim,
    # input_dim=data_spec.n_actions,
    input_dim=data_spec.n_actions + ll_model_params.nz_vae + ll_model_params.n_rollout_steps,
    output_dim=1,
    n_layers=5,  # number of policy network layer
    nz_mid=256,
    action_input=True,
    # unused_obs_size = ll_model_params.nz_vae + ll_model_params.n_rollout_steps, # whether remove latent variable, or add it
)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
ll_agent_config = AttrDict(
    # policy=CDModelPolicy,
    policy=TimeIndexedCDMdlPolicy,
    policy_params=ll_policy_params,
    critic=MLPCritic,                   # LL critic is not used since we are not finetuning LL
    # critic=SplitObsMLPCritic,
    critic_params=ll_critic_params
)

# update HL policy model params
hl_policy_params.update(AttrDict(
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
))

# register new LL agent in agent_config and turn off LL agent updates
agent_config.update(AttrDict(
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    update_ll=True,
))