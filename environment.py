"""
Using models cited in
 'Power law models of infectious disease spread'
  https://projecteuclid.org/download/pdfview_1/euclid.aoas/1414091227
"""
import pdb
import numpy as np


class PoissonDisease(object):
    """
    Adapted from estimates in 'Modeling seasonality in space-time infectious disease surveillance',
    pdf pg. 9
    https://onlinelibrary.wiley.com/doi/pdf/10.1002/bimj.201200037?casa_token=rT0pGljzSzMAAAAA:8R7hmAW4N4box2PqLqlPLEbOKdQPTs4dIZ34McLZQJQPuPf-uEC5rA_C4ljzfUQYVGt6yz7rnKKYKrQ

    In their notation, our beta_. is given by beta_. = [gamma^., delta^.]
    Assuming seasonal variation only in endemic component.
    """

    def __init__(self, L, lambda_a=0.2, phi_a=0.1, alpha_nu=-3.12, alpha_lambda=-0.67, alpha_phi=-1.01,
                 beta_nu=np.array([1.91, 2.69]), kernel_bandwidth=0.1):
        self.L = L
        self.alpha_nu = alpha_nu
        self.alpha_lambda = alpha_lambda
        self.alpha_phi = alpha_phi
        self.lambda_a = lambda_a
        self.phi_a = phi_a
        self.beta_nu = beta_nu
        self.omega = 2 * np.pi / 52

        # lambda (autoregressive effect)
        self.lambda_ = np.exp(self.alpha_lambda)

        # phi (spatiotemporal effect)
        self.phi = np.exp(self.alpha_phi)

        # Generate coordinates uniformly on root_L x root_L square
        root_L = np.sqrt(L)
        coordinates = np.random.uniform(low=0, high=root_L, size=L)

        # Construct spatial weight matrix
        self.spatial_weight_matrix = np.zeros((L, L))
        for i in range(L):
            for j in range(i, L):
                coord_i = coordinates[i]
                coord_j = coordinates[j]
                d_ij = np.abs(coord_i - coord_j).sum()
                weight_ij = np.exp(-d_ij / kernel_bandwidth)
                self.spatial_weight_matrix[i, j] = weight_ij
                self.spatial_weight_matrix[j, i] = weight_ij

        self.Y = None
        self.t = None
        self.X_list = []  # For collecting dependent variables into sequence of design matrices
        self.Y_list = []

    def get_endemic_effect(self, t):
        z = np.array([np.sin(self.omega * t), np.cos(self.omega * t)])
        log_nu = self.alpha_nu + np.dot(self.beta_nu, z)
        nu = np.exp(log_nu)
        z = [z for _ in range(self.L)]
        return nu, z

    def reset(self):
        # ToDo: option to reset coordinates?

        self.Y = np.random.poisson(1, size=self.L)
        self.t = 1
        self.X_list = []
        self.Y_list = []

    def mean_counts(self, Ytm1, Atm1, nu):
        endemic = nu
        action_infection_interaction = Ytm1 * Atm1
        autoregressive = self.lambda_ * Ytm1
        autoregressive_action = self.lambda_a * action_infection_interaction
        spatial_weight_times_ytm1 = np.dot(self.spatial_weight_matrix, Ytm1)
        spatiotemporal = self.phi * spatial_weight_times_ytm1
        spatial_weight_times_interaction = np.dot(self.spatial_weight_matrix, action_infection_interaction)
        spatiotemporal_action = self.phi_a * spatial_weight_times_interaction
        mean_counts_ = endemic + autoregressive - autoregressive_action + spatiotemporal - spatiotemporal_action
        mean_counts_ = np.maximum(mean_counts_, 0)
        return mean_counts_, action_infection_interaction, spatial_weight_times_ytm1, spatial_weight_times_interaction

    def step(self, A):

        nu, z = self.get_endemic_effect(self.t)
        mean_counts_, action_infection_interaction, spatial_weight_times_ytm1, spatial_weight_times_interaction = \
            self.mean_counts(self.Y, A, nu)
        X_t = np.column_stack((np.ones(self.L), z, self.Y, action_infection_interaction, spatial_weight_times_ytm1,
                               spatial_weight_times_interaction))
        Y = np.random.poisson(mean_counts_)
        self.Y = Y
        self.t += 1
        self.X_list.append(X_t)
        self.Y_list.append(Y)
        return Y


if __name__ == "__main__":
    L = 50
    env = PoissonDisease(L=L)

