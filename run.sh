#!/usr/bin/env bash

python3 run.py --policy_name='myopic_model_free' --L=15 --time_horizon=25 --budget=10 --num_replicates=1 \
               --true_kernel='network' --specified_kernel='global' --global_kernel_bandwidth=10 \
               --replicate_batches=1
