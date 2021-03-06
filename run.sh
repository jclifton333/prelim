#!/usr/bin/env bash

# python3 run.py --policy_name='random' --L=50 --time_horizon=25 --budget=10 --num_replicates=192 \
#                --true_kernel='global' --specified_kernel='global' --global_kernel_bandwidth=3.0 \
#                --replicate_batches=8

python3 run.py --policy_name='treat_all' --L=50 --time_horizon=25 --budget=10 --num_replicates=192 \
               --true_kernel='global' --specified_kernel='global' --global_kernel_bandwidth=3.0 \
               --replicate_batches=8

python3 run.py --policy_name='treat_none' --L=50 --time_horizon=25 --budget=10 --num_replicates=192 \
               --true_kernel='global' --specified_kernel='global' --global_kernel_bandwidth=3.0 \
               --replicate_batches=8

python3 run.py --policy_name='policy_search' --L=50 --time_horizon=25 --budget=10 --num_replicates=192 \
               --true_kernel='global' --specified_kernel='global' --global_kernel_bandwidth=3.0 \
               --replicate_batches=8

python3 run.py --policy_name='policy_search' --L=50 --time_horizon=25 --budget=10 --num_replicates=192 \
               --true_kernel='global' --specified_kernel='network' --global_kernel_bandwidth=3.0 \
               --replicate_batches=8

python3 run.py --policy_name='one_step_fitted_q' --L=50 --time_horizon=25 --budget=10 --num_replicates=192 \
               --true_kernel='global' --specified_kernel='global' --global_kernel_bandwidth=3.0 \
               --replicate_batches=8

python3 run.py --policy_name='one_step_fitted_q' --L=50 --time_horizon=25 --budget=10 --num_replicates=192 \
               --true_kernel='global' --specified_kernel='network' --global_kernel_bandwidth=3.0 \
               --replicate_batches=8

python3 run.py --policy_name='myopic_model_based' --L=50 --time_horizon=25 --budget=10 --num_replicates=192 \
               --true_kernel='global' --specified_kernel='global' --global_kernel_bandwidth=3.0 \
               --replicate_batches=8

python3 run.py --policy_name='myopic_model_based' --L=50 --time_horizon=25 --budget=10 --num_replicates=192 \
               --true_kernel='global' --specified_kernel='network' --global_kernel_bandwidth=3.0 \
               --replicate_batches=8

python3 run.py --policy_name='myopic_model_free' --L=50 --time_horizon=25 --budget=10 --num_replicates=192 \
               --true_kernel='global' --specified_kernel='global' --global_kernel_bandwidth=3.0 \
               --replicate_batches=8

python3 run.py --policy_name='myopic_model_free' --L=50 --time_horizon=25 --budget=10 --num_replicates=192 \
               --true_kernel='global' --specified_kernel='network' --global_kernel_bandwidth=3.0 \
               --replicate_batches=8

