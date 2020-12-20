#!/usr/bin/env bash

python3 run.py --policy_name='random' --L=50 --time_horizon=20 --budget=15 --num_replicates=2

python3 run.py --policy_name='myopic_model_free' --L=50 --time_horizon=20 --budget=15 --num_replicates=2

python3 run.py --policy_name='one_step_fitted_q' --L=50 --time_horizon=20 --budget=15 --num_replicates=2
