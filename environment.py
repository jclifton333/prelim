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
                 beta_nu=np.array([1.91, 2.69]), kernel_bandwidth=1):
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

        # Construct spatial weight matrix
        self.spatial_weight_matrix = np.zeros((L, L))
        for i in range(L):
            for j in range(i, L):
                d_ij = np.abs(i - j)
                weight_ij = np.exp(-d_ij / kernel_bandwidth)
                self.spatial_weight_matrix[i, j] = weight_ij
                self.spatial_weight_matrix[j, i] = weight_ij

        self.Y = None
        self.t = None

    def get_endemic_effect(self, t):
        z = np.array([np.sin(self.omega * t), np.cos(self.omega * t)])
        log_nu = self.alpha_nu + np.dot(self.beta_nu, z)
        nu = np.exp(log_nu)

        return nu

    def reset(self):
        self.Y = np.random.poisson(1, size=self.L)
        self.t = 1

    def mean_counts(self, Ytm1, Atm1, nu):
        endemic = nu
        action_infection_interaction = Ytm1 * Atm1
        autoregressive = self.lambda_ * Ytm1
        autoregressive_action = self.lambda_a * action_infection_interaction
        spatiotemporal = self.phi * np.dot(self.spatial_weight_matrix, Ytm1)
        spatiotemporal_action = self.phi_a * np.dot(self.spatial_weight_matrix, action_infection_interaction)
        mean_counts_ = endemic + autoregressive - autoregressive_action + spatiotemporal - spatiotemporal_action
        mean_counts_ = np.maximum(mean_counts_, 0)
        return mean_counts_

    def step(self, A):
        nu = self.get_endemic_effect(self.t)
        mean_counts_ = self.mean_counts(self.Y, A, nu)
        Y = np.random.poisson(mean_counts_)
        self.Y = Y
        self.t += 1
        return Y


if __name__ == "__main__":
    L = 50
    env = PoissonDisease(L=L)

