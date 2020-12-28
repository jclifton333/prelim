#!/usr/bin/env bash

python3 run.py --policy_name='random' --L=50 --time_horizon=5 --budget=5 --num_replicates=24 \
               --true_kernel='network' --specified_kernel='network'

python3 run.py --policy_name='treat_all' --L=50 --time_horizon=25 --budget=10 --num_replicates=24 \
               --true_kernel='network' --specified_kernel='network'

python3 run.py --policy_name='treat_none' --L=50 --time_horizon=25 --budget=10 --num_replicates=24 \
               --true_kernel='network' --specified_kernel='network'

python3 run.py --policy_name='myopic_model_free' --L=50 --time_horizon=25 --budget=10 --num_replicates=24 \
               --true_kernel='network' --specified_kernel='network'

python3 run.py --policy_name='myopic_model_free' --L=50 --time_horizon=25 --budget=10 --num_replicates=24 \
               --true_kernel='network' --specified_kernel='global'

python3 run.py --policy_name='oracle_greedy_model_based' --L=50 --time_horizon=25 --budget=10 --num_replicates=24 \
               --true_kernel='network' --specified_kernel='network'

python3 run.py --policy_name='oracle_greedy_model_based' --L=50 --time_horizon=25 --budget=10 --num_replicates=24 \
               --true_kernel='network' --specified_kernel='global'

python3 run.py --policy_name='greedy_model_based' --L=50 --time_horizon=25 --budget=10 --num_replicates=24 \
               --true_kernel='network' --specified_kernel='network'

python3 run.py --policy_name='greedy_model_based' --L=50 --time_horizon=25 --budget=10 --num_replicates=24 \
               --true_kernel='network' --specified_kernel='global'

python3 run.py --policy_name='oracle_policy_search' --L=50 --time_horizon=25 --budget=10 --num_replicates=24 \
               --true_kernel='network' --specified_kernel='network'

python3 run.py --policy_name='oracle_policy_search' --L=50 --time_horizon=25 --budget=10 --num_replicates=24 \
               --true_kernel='network' --specified_kernel='global'

python3 run.py --policy_name='policy_search' --L=50 --time_horizon=25 --budget=10 --num_replicates=24 \
               --true_kernel='network' --specified_kernel='network'

python3 run.py --policy_name='policy_search' --L=50 --time_horizon=25 --budget=10 --num_replicates=24 \
               --true_kernel='network' --specified_kernel='global'
