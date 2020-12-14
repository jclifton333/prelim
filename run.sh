#!/usr/bin/env bash

python3 run.py --policy_name='random' --L=50 --time_horizon=25 --budget=15 --num_replicates=10

python3 run.py --policy_name='treat_none' --L=50 --time_horizon=25 --budget=15 --num_replicates=10

python3 run.py --policy_name='treat_all' --L=50 --time_horizon=25 --budget=15 --num_replicates=10

# python3 run.py --policy_name='greedy_model_based' --L=50 --time_horizon=25 --budget=15 --num_replicates=10

python3 run.py --policy_name='policy_search' --L=50 --time_horizon=25 --budget=15 --num_replicates=10