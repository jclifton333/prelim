This repository implements the experiments described in my prelim Paper 2. The generative model used 
in the experiments is based on that of Held and Paul (2012), "Modeling seasonality in space-time infectious
disease data".  


## Contents

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

## Notes

- The `network` and `global` kernels in this codebase correspond respectively to \kappa_1 and \kappa_\delta 
(for user-specified \delta) in Paper 2.
- The notation for the parameters of the generative model differs between Paper 2 and this code. In the
paper, these are referred to as \beta = [\beta_0, ..., \beta_6]. In the code, they are referred to as 
`model_parameters = [alpha_nu, beta_1, beta_2, lambda, lambda_a, phi, phi_a]`. This is based on 
Held and Paul (2012)'s notation.
