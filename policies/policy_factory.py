from . import baseline_policies
from . import fitted_q
from . import policy_search


def policy_factory(policy_name):
    if policy_name == 'policy_search':
        return policy_search.policy_search_policy
    elif policy_name == 'random':
        return baseline_policies.random_policy
    elif policy_name == 'treat_all':
        return baseline_policies.treat_all_policy
    elif policy_name == 'treat_none':
        return baseline_policies.treat_none_policy
    elif policy_name == 'greedy_model_based':
        return baseline_policies.greedy_model_based_policy
    elif policy_name == 'oracle_greedy_model_based':
        return baseline_policies.oracle_greedy_model_based_policy
    elif policy_name == 'myopic_model_free':
        return fitted_q.myopic_model_free_policy
    elif policy_name == 'one_step_fitted_q':
        return fitted_q.one_step_fitted_q_policy
    elif policy_name == 'oracle_policy_search':
        return policy_search.oracle_policy_search_policy
    elif policy_name == 'greedy_model_free':
        return fitted_q.greedy_model_free_policy
    elif policy_name == 'myopic_model_based':
        return fitted_q.myopic_model_based_policy
    elif policy_name == 'oracle_one_step_fitted_q':
        return fitted_q.oracle_one_step_fitted_q
