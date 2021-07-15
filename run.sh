#!/usr/bin/env bash

python3 run.py --policy_name='policy_search' --L=50 --time_horizon=10 --budget=10 --num_replicates=192 \
               --true_kernel='global' --specified_kernel='global' --global_kernel_bandwidth=3.0 \
               --replicate_batches=8 --policies_to_compare='policy_search,myopic_model_based'

python3 run.py --policy_name='myopic_model_based' --L=50 --time_horizon=10 --budget=10 --num_replicates=192 \
               --true_kernel='global' --specified_kernel='global' --global_kernel_bandwidth=3.0 \
               --replicate_batches=8 --policies_to_compare='policy_search,myopic_model_based'

python3 run.py --policy_name='mbm' --L=50 --time_horizon=10 --budget=10 --num_replicates=192 \
               --true_kernel='global' --specified_kernel='global' --global_kernel_bandwidth=3.0 \
               --replicate_batches=8 --policies_to_compare='policy_search,myopic_model_based'

