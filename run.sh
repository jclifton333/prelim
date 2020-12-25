#!/usr/bin/env bash

python3 run.py --policy_name='oracle_greedy_model_based' --L=50 --time_horizon=50 --budget=10 --num_replicates=24

python3 run.py --policy_name='oracle_greedy_model_based' --L=100 --time_horizon=50 --budget=20 --num_replicates=24

python3 run.py --policy_name='oracle_greedy_model_based' --L=200 --time_horizon=50 --budget=40 --num_replicates=24


