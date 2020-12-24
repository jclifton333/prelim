#!/usr/bin/env bash

python3 -m cProfile -o profile_output run.py --policy_name='policy_search' --L=10 --time_horizon=5 --budget=2 --num_replicates=1
python3 -m cprofilev -f profile_output