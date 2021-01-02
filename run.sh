#!/usr/bin/env bash

python3 run.py --policy_name='myopic_model_based' --L=50 --time_horizon=25 --budget=10 --num_replicates=10 \
               --true_kernel='network' --specified_kernel='network' --global_kernel_bandwidth=10 \
               --replicate_batches=5

