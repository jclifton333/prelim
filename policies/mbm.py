import numpy as np
from . import policy_search as ps
from .model_estimation import fit_model
from .policy_factory import policy_factory
import pdb

BURN_IN = 5
BURN_IN_POLICY = policy_factory('random')


def roll_out_policies(policies_to_compare, model_parameter_posterior, env, budget, time_horizon, kernel,
                      discount_factor=0.96, n_posterior_reps=10):

  # ToDo: assuming kernel specified for policy and kernel for rollout env are the same
  spatial_weight_matrices = env.get_spatial_weight_matrices()  # ToDo: what assumptions about model does this make?
  scores_dict = {policy_name: np.zeros(n_posterior_reps) for policy_name in policies_to_compare}
  policies_dict = {policy_name: policy_factory(policy_name) for policy_name in policies_to_compare}

  # Estimate posterior expected value of each policy
  for posterior_rep in range(n_posterior_reps):
    model_parameter = model_parameter_posterior()
    rollout_env = ps.env_from_model_parameter(model_parameter, env.Y, env.t, env.L, kernel, spatial_weight_matrices)
    for policy_name in policies_to_compare:
      # ToDo: modify PoissonDisease so that resets to initial X, if initial X is passed (and also policy search?)
      rollout_env.reset()
      policy_ = policies_dict[policy_name]

      # Burn in
      for t in range(BURN_IN):
        action_info = BURN_IN_POLICY(rollout_env, budget, time_horizon-t, discount_factor)
        rollout_env.step(action_info['A'])

      # Simulate episode under this policy and model parameter
      for t in range(time_horizon):  # ToDo: start at beginning or current timestep?
        # Roll out this policy
        action_info = policy_(rollout_env, budget, time_horizon-t, discount_factor, kernel=kernel)
        rollout_env.step(action_info['A'])
        scores_dict[policy_name][posterior_rep] += discount_factor**t + rollout_env.Y.sum()

  # Get the best policy
  best_score = float('inf')
  best_policy = None
  for k, v in scores_dict.items():
    score = v.mean()
    if score < best_score:
      best_policy = policies_dict[k]
      best_score = score

  print('got here')
  # Get action from best policy
  action_info = best_policy(env, budget, time_horizon, discount_factor, kernel=kernel)
  return action_info


def mbm_policy(env, budget, time_horizon, discount_factor, kernel='network',
               policies_to_compare=('greedy_model_free', 'myopic_model_free')):

  def model_parameter_posterior():
    sampled_model_parameters = fit_model(env, perturb=True, kernel=kernel)  # ToDo: check that this is the right kernel
    return sampled_model_parameters

  action_info = roll_out_policies(policies_to_compare, model_parameter_posterior, env, budget, time_horizon,
                                  kernel, discount_factor=discount_factor)

  return action_info

