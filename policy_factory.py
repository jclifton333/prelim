from policy_search import policy_search_policy
import baseline_policies
import fitted_q


def policy_factory(policy_name):
    if policy_name == 'policy_search':
        return policy_search_policy
    elif policy_name == 'random':
        return baseline_policies.random_policy
    elif policy_name == 'treat_all':
        return baseline_policies.treat_all_policy
    elif policy_name == 'treat_none':
        return baseline_policies.treat_none_policy
    elif policy_name == 'greedy_model_based':
        return baseline_policies.greedy_model_based_policy
    elif policy_name == 'myopic_model_free':
        return fitted_q.myopic_model_free_policy
    elif policy_name == 'one_step_fitted_q':
        return fitted_q.one_step_fitted_q_policy
    elif policy_name == 'one_step_fitted_q_propensity':
        return fitted_q.one_step_fitted_q_propensity_policy