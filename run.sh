#!/usr/bin/env bash

python3 run.py --policy_name='policy_search' --L=50 --time_horizon=25 --budget=10 --num_replicates=20 \
               --true_kernel='network' --specified_kernel='global' --global_kernel_bandwidth=50 \
               --replicate_batches=5

python3 run.py --policy_name='policy_search' --L=50 --time_horizon=25 --budget=10 --num_replicates=20 \
               --true_kernel='network' --specified_kernel='network' --global_kernel_bandwidth=50 \
               --replicate_batches=5

python3 run.py --policy_name='random' --L=50 --time_horizon=25 --budget=10 --num_replicates=20 \
               --true_kernel='network' --specified_kernel='network' --global_kernel_bandwidth=50 \
               --replicate_batches=5