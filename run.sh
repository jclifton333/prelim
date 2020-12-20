#!/usr/bin/env bash

python3 run.py --policy_name='random' --L=50 --time_horizon=25 --budget=10 --num_replicates=24

python3 run.py --policy_name='treat_all' --L=50 --time_horizon=25 --budget=10 --num_replicates=24

python3 run.py --policy_name='treat_none' --L=50 --time_horizon=25 --budget=10 --num_replicates=24

python3 run.py --policy_name='greedy_model_based' --L=50 --time_horizon=25 --budget=10 --num_replicates=24

python3 run.py --policy_name='myopic_model_free' --L=50 --time_horizon=25 --budget=10 --num_replicates=24

python3 run.py --policy_name='one_step_fitted_q' --L=50 --time_horizon=25 --budget=10 --num_replicates=24

python3 run.py --policy_name='policy_search' --L=50 --time_horizon=25 --budget=10 --num_replicates=24

python3 run.py --policy_name='random' --L=100 --time_horizon=25 --budget=20 --num_replicates=24

python3 run.py --policy_name='treat_all' --L=100 --time_horizon=25 --budget=20 --num_replicates=24

python3 run.py --policy_name='treat_none' --L=100 --time_horizon=25 --budget=20 --num_replicates=24

python3 run.py --policy_name='greedy_model_based' --L=100 --time_horizon=25 --budget=20 --num_replicates=24

python3 run.py --policy_name='myopic_model_free' --L=100 --time_horizon=25 --budget=20 --num_replicates=24

python3 run.py --policy_name='one_step_fitted_q' --L=100 --time_horizon=25 --budget=20 --num_replicates=24

python3 run.py --policy_name='policy_search' --L=100 --time_horizon=25 --budget=20 --num_replicates=24

python3 run.py --policy_name='random' --L=200 --time_horizon=25 --budget=40 --num_replicates=24

python3 run.py --policy_name='treat_all' --L=200 --time_horizon=25 --budget=40 --num_replicates=24

python3 run.py --policy_name='treat_none' --L=200 --time_horizon=25 --budget=40 --num_replicates=24

python3 run.py --policy_name='greedy_model_based' --L=200 --time_horizon=25 --budget=40 --num_replicates=24

python3 run.py --policy_name='myopic_model_free' --L=200 --time_horizon=25 --budget=40 --num_replicates=24

python3 run.py --policy_name='one_step_fitted_q' --L=200 --time_horizon=25 --budget=40 --num_replicates=24

python3 run.py --policy_name='policy_search' --L=200 --time_horizon=25 --budget=40 --num_replicates=24

