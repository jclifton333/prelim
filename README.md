This repository implements the experiments described in my prelim Paper 2. 

- `environment`: Contains the `PoissonDisease` class, which implements the generative model with which policies will  
interact; 
- `policies`: Contains scripts for
    - `fitted_q`: Model-free myopic and one-step fitted-Q policies, as well as oracle one-step fitted-Q; 
    - `policy_search`: Model-based policy search and oracle policy search;
    - `model_estimation`: Implements penalized maximum-likelihood estimation of the disease process model; 
    - `baseline_policies`: Random, treat-all, treat-none, and greedy policies; 
    - `policy_factory`: Function for getting policy by string.
- `optim`: Code for optimizing Q-functions and policies  
    - `optim`: Main functions for optimization; 
    - `linear_relaxation`: Fit and solve linear approximation to Q-function optimization problem;
- `run`: Run experiments based on user-specified settings.  