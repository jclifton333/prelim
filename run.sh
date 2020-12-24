#!/usr/bin/env bash

python3 run.py --policy_name='oracle_policy_search' --L=50 --time_horizon=50 --budget=10 --num_replicates=24

python3 run.py --policy_name='oracle_policy_search' --L=100 --time_horizon=50 --budget=10 --num_replicates=24

python3 run.py --policy_name='oracle_policy_search' --L=200 --time_horizon=50 --budget=10 --num_replicates=24


