"""
Using models cited in
 'Power law models of infectious disease spread'
  https://projecteuclid.org/download/pdfview_1/euclid.aoas/1414091227
"""
import numpy as np
OMEGA = 2*np.pi / 52


def time_varying_params(t, alpha, b_lambda, b_phi, b_nu, beta_lambda, beta_phi, beta_nu):

    z = np.array([t, np.sin(OMEGA * t)])

    # nu (endemic effect)
    log_nu = alpha + b_nu + np.dot(beta_nu, z)
    nu = np.exp(log_nu)

    # lambda (autoregressive effect)
    log_lambda = alpha + b_lambda + np.dot(beta_lambda, z)
    lambda_ = np.exp(log_lambda)

    # phi (spatiotemporal effect)
    log_phi = alpha + b_phi + np.dot(beta_phi, z)
    phi = np.exp(log_phi)

    return nu, lambda_, phi


def mean_counts(Ytm1, Atm1, nu, e, lambda_, phi, lambda_a, phi_a, spatial_weight_matrix):

    endemic = nu * e
    autoregressive = lambda_ * Ytm1
    autoregressive_action = lambda_a * Atm1
    spatiotemporal = phi * np.dot(spatial_weight_matrix, Ytm1)
    spatiotemporal_action = phi_a * np.dot(spatial_weight_matrix, Atm1)
    mean_counts_ = endemic + autoregressive + autoregressive_action + spatiotemporal + spatiotemporal_action
    return mean_counts_


def step(Ytm1, Atm1, t, e, alpha, b_lambda, b_phi, b_nu, beta_lambda, beta_phi, beta_nu, lambda_a,
         phi_a, spatial_weight_matrix):
    nu, lambda_, phi = time_varying_params(t, alpha, b_lambda, b_phi, b_nu, beta_lambda, beta_phi, beta_nu)
    mean_counts_ = mean_counts(Ytm1, Atm1, nu, e, lambda_, phi, lambda_a, phi_a, spatial_weight_matrix)
    Y = np.random.poisson(mean_counts_)
    return Y




