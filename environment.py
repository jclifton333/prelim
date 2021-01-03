"""
Using models cited in
 'Power law models of infectious disease spread'
  https://projecteuclid.org/download/pdfview_1/euclid.aoas/1414091227
"""
import pdb
import numpy as np
import copy


class PoissonDisease(object):
    """
    Adapted from estimates in 'Modeling seasonality in space-time infectious disease surveillance',
    pdf pg. 9
    https://drive.google.com/file/d/1P1McRZng8sNoXRlrJK--08Jnv4vsKZcV/view?usp=sharing

    In their notation, our beta_. is given by beta_. = [gamma^., delta^.]
    Assuming seasonal variation only in endemic component.
    """

    def __init__(self, L, lambda_a=0.2, phi_a=0.1, alpha_nu=-3.12, alpha_lambda=-0.67, alpha_phi=-1.01,
                 beta_nu=np.array([1.91, 2.69]), kernel_bandwidth=1, Y_initial=None,
                 t_initial=None, kernel='network', spatial_weight_matrices=None):

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

        # Construct spatial weight matrix covariance matrix
        if spatial_weight_matrices is None:
            self.network_spatial_weight_matrix = np.zeros((L, L))
            self.global_spatial_weight_matrix = np.zeros((L, L))
            for i in range(L):
                for j in range(i, L):
                    coord_i = coordinates[i]
                    coord_j = coordinates[j]

                    # Get network kernel
                    d_ij = np.abs(coord_i - coord_j).sum()
                    weight_ij = d_ij < 1
                    self.network_spatial_weight_matrix[i, j] = weight_ij
                    self.network_spatial_weight_matrix[j, i] = weight_ij

                    # Get global kernel
                    # global_weight_ij = np.exp(-d_ij / kernel_bandwidth)
                    global_weight_ij = 1 / (1 + d_ij/kernel_bandwidth)
                    self.global_spatial_weight_matrix[i, j] = global_weight_ij
                    self.global_spatial_weight_matrix[j, i] = global_weight_ij

            self.network_spatial_weight_matrix /= self.network_spatial_weight_matrix.sum(axis=1)
            self.global_spatial_weight_matrix /= self.global_spatial_weight_matrix.sum(axis=1)
        else:
            self.set_spatial_weight_matrices(spatial_weight_matrices)

        self.kernel = kernel
        if kernel == 'network':
            self.spatial_weight_matrix = self.network_spatial_weight_matrix
        else:
            self.spatial_weight_matrix = self.global_spatial_weight_matrix

        self.Y_initial = Y_initial
        self.t_initial = t_initial

        self.Y = None
        self.X = None
        self.K = None
        self.K_global = None
        self.K_network = None
        self.t = None
        self.X_list = []  # For collecting dependent variables into sequence of design matrices
        self.Y_list = []
        self.K_network_list = []
        self.K_global_list = []
        self.K_list = []
        self.propensities_list = []

    def set_spatial_weight_matrices(self, spatial_weight_matrices):
        self.network_spatial_weight_matrix = spatial_weight_matrices['network_spatial_weight_matrix']
        self.global_spatial_weight_matrix = spatial_weight_matrices['global_spatial_weight_matrix']

    def get_spatial_weight_matrices(self):
        spatial_weight_matrices = {'network_spatial_weight_matrix': self.network_spatial_weight_matrix,
                                   'global_spatial_weight_matrix': self.global_spatial_weight_matrix}
        return spatial_weight_matrices

    def get_endemic_effect(self, t):
        z = np.array([np.sin(self.omega * t), np.cos(self.omega * t)])
        log_nu = self.alpha_nu + np.dot(self.beta_nu, z)
        nu = np.exp(log_nu)
        z = [z for _ in range(self.L)]
        return nu, z

    def reset(self):
        # ToDo: option to reset coordinates?

        if self.Y_initial is None:
            self.Y = np.random.poisson(1, size=self.L)
        else:
            self.Y = self.Y_initial

        if self.t_initial is None:
            self.t = 1
        else:
            self.t = self.t_initial

        self.K = None
        self.K_global = None
        self.K_network = None
        self.X_list = []
        self.Y_list = []
        self.K_network_list = []
        self.K_global_list = []
        self.K_list = []
        self.propensities_list = []

    def get_X_at_A(self, X, K, A, kernel=None):
        # ToDo: return separate arrays for kernel and non-kernel features

        Y = X[:, 3]
        action_infection_interaction = Y * A

        if kernel is None:
            spatial_weight_times_interaction = np.dot(self.spatial_weight_matrix, action_infection_interaction)
        elif kernel == 'network':
            spatial_weight_times_interaction = np.dot(self.network_spatial_weight_matrix, action_infection_interaction)
        elif kernel == 'global':
            spatial_weight_times_interaction = np.dot(self.global_spatial_weight_matrix, action_infection_interaction)

        X_new = copy.copy(X)
        K_new = copy.copy(K)
        X_new[:, 4] = -action_infection_interaction
        K_new[:, 1] = -spatial_weight_times_interaction
        return X_new, K_new

    def get_kernel_terms(self, A, Y):
        K = self.get_K(A, Y, kernel='true')
        K_global = self.get_K(A, Y, kernel='global')
        K_network = self.get_K(A, Y, kernel='network')
        return K_global, K_network, K

    def get_K_history(self, kernel):
        if kernel == 'network':
            K_list = self.K_network_list
        elif kernel == 'global':
            K_list = self.K_global_list
        elif kernel == 'true':
            K_list = self.K_list
        return K_list

    def get_current_K(self, kernel):
        if kernel == 'network':
            K = self.K_network
        elif kernel == 'global':
            K = self.K_global
        elif kernel == 'true':
            K = self.K
        return K

    def get_features_for_mean(self, Ytm1, Atm1, kernel='true'):
        spatial_weight_matrix = self.get_spatial_weight_matrix(kernel=kernel)
        endemic, z = self.get_endemic_effect(self.t)
        action_infection_interaction = Ytm1 * Atm1
        autoregressive = self.lambda_ * Ytm1
        autoregressive_action = self.lambda_a * action_infection_interaction
        spatial_weight_times_ytm1 = np.dot(spatial_weight_matrix, Ytm1)
        spatiotemporal = self.phi * spatial_weight_times_ytm1
        spatial_weight_times_interaction = np.dot(spatial_weight_matrix, action_infection_interaction)
        spatiotemporal_action = self.phi_a * spatial_weight_times_interaction
        return z, endemic, autoregressive, -autoregressive_action, spatiotemporal, -spatiotemporal_action

    def mean_counts(self, Ytm1, Atm1):
        z, endemic, autoregressive, autoregressive_action, spatiotemporal, spatiotemporal_action = \
            self.get_features_for_mean(Ytm1, Atm1)
        mean_counts_ = endemic + autoregressive + autoregressive_action + spatiotemporal + spatiotemporal_action
        mean_counts_ = np.maximum(mean_counts_, 0)
        return mean_counts_

    def get_spatial_weight_matrix(self, kernel):
        if kernel == 'true':
            return self.spatial_weight_matrix
        elif kernel == 'network':
            return self.network_spatial_weight_matrix
        elif kernel == 'global':
            return self.global_spatial_weight_matrix

    def get_X(self, A, Y, t):
        _, z = self.get_endemic_effect(t)
        action_infection_interaction = np.multiply(A, Y)
        X = np.column_stack((np.ones(self.L), z, Y, -action_infection_interaction))
        return X

    def get_K(self, A, Y, kernel='true'):
        action_infection_interaction = np.multiply(A, Y)
        spatial_weight_matrix = self.get_spatial_weight_matrix(kernel=kernel)
        spatial_weight_times_y = np.dot(spatial_weight_matrix, Y)
        spatial_weight_times_interaction = np.dot(spatial_weight_matrix, action_infection_interaction)
        K = np.column_stack((spatial_weight_times_y, spatial_weight_times_interaction))
        return K

    def draw_next_state(self, A, kernel='true'):
        # ToDo: this can be optimized since the only thing that changes between draws is Ytp
        mean_counts_ = self.mean_counts(self.Y, A)
        Ytp = np.random.poisson(mean_counts_)
        Xtp = self.get_X(A, Ytp, self.t+1)
        Ktp = self.get_K(A, Ytp, kernel=kernel)
        return Xtp, Ktp

    def step(self, A, propensities=None):
        mean_counts_ = self.mean_counts(self.Y, A)
        X = self.get_X(A, self.Y, self.t)
        Y = np.random.poisson(mean_counts_)
        K_global, K_network, K = self.get_kernel_terms(A, self.Y)

        # Update current states
        self.X = X
        self.Y = Y
        self.K = K
        self.K_global = K_global
        self.K_network = K_network
        self.t += 1

        # Update histories
        self.K_network_list.append(K_network)
        self.K_global_list.append(K_global)
        self.K_list.append(K)
        self.X_list.append(X)
        self.Y_list.append(Y)
        if propensities is not None:
            self.propensities_list.append(propensities)
        return Y


if __name__ == "__main__":
    L = 50
    env = PoissonDisease(L=L)

