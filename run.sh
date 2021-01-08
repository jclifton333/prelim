#!/usr/bin/env bash

python3 run.py --policy_name='random' --L=20 --time_horizon=25 --budget=10 --num_replicates=10 \
               --true_kernel='network' --specified_kernel='network' --global_kernel_bandwidth=10 \
               --replicate_batches=5

python3 run.py --policy_name='myopic_model_free' --L=20 --time_horizon=25 --budget=10 --num_replicates=10 \
               --true_kernel='network' --specified_kernel='global' --global_kernel_bandwidth=10 \
               --replicate_batches=5

