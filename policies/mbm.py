import numpy as np
from policy_factory import policy_factory
import policy_search as ps


def roll_out_policies(policies_to_compare, model_parameter_posterior, env, budget, time_horizon, kernel,
                      discount_factor=0.96, n_posterior_reps=10):

  # ToDo: assuming kernel specified for policy and kernel for rollout env are the same
  spatial_weight_matrices = env.get_spatial_weight_matrices()  # ToDo: what assumptions about model does this make?
  policies_dict = {policy_name: policy_factory(policy_name) for policy_name in policies_to_compare}
  scores_dict = {policy_name: np.zeros(n_posterior_reps) for policy_name in policies_to_compare}

  # Estimate posterior expected value of each policy
  for posterior_rep in range(n_posterior_reps):
    model_parameter = model_parameter_posterior()
    rollout_env = ps.env_from_model_parameter(model_parameter, env.Y, env.t, env.L, kernel, spatial_weight_matrices)

    for policy_name in policies_to_compare:
      rollout_env.reset()
      policy_ = policies_dict[policy_name]

      for t in range(time_horizon - env.t):  # ToDo: start at beginning or current timestep?
        # Roll out this policy
        action_info = policy_(rollout_env, budget, time_horizon, discount_factor, kernel=kernel)
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

  # Get action from best policy
  action_info = best_policy(env, budget, time_horizon, discount_factor, kernel=kernel)
  return action_info
