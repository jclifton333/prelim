#!/usr/bin/env bash

python3 run.py --policy_name='oracle_one_step_fitted_q' --L=50 --time_horizon=25 --budget=10 --num_replicates=24 \
               --true_kernel='network' --specified_kernel='network' --global_kernel_bandwidth=10 \
               --replicate_batches=1

python3 run.py --policy_name='one_step_fitted_q' --L=50 --time_horizon=25 --budget=10 --num_replicates=24 \
               --true_kernel='network' --specified_kernel='network' --global_kernel_bandwidth=10 \
               --replicate_batches=1