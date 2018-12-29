"""
For estimating the parameters of a 2d-Gaussian variable using Maximum a posteriori with an normal-inverse-Wishart prior.
Useful lines to include for editing:
    execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
"""
import os
execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import multivariate_normal, invwishart
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser(description='Demonstrate Bayesian inference for the parameters of a 2-D Gaussian distribution.')
parser.add_argument('-m', '--true_mean', help='Mean of the true distn.', type=float, nargs=2, default=[1.2, -2.3])
parser.add_argument('-v', '--true_variance', help='Variance of the true distn.', type=float, nargs=4, default=[[1.0, 0.0], [0.0, 1.0]])
parser.add_argument('-o', '--prior_mean', help='Mean of the prior distn.', type=float, nargs=2, default=[0.0, 0.0])
parser.add_argument('-p', '--prior_variance', help='Variance of the prior distn.', type=float, nargs=4, default=[[0.1, 0.0], [0.0, 0.1]])
parser.add_argument('-k', '--prior_kappa', help='Kappa parameter of the prior distn.', type=int, default=1)
parser.add_argument('-u', '--prior_nu', help='Nu parameter of the prior distn.', type=int, default=3)
parser.add_argument('-n', '--num_data_points', help='The number of data points to sample from true distn.', type=int, default=100)
parser.add_argument('-d', '--debug', help='Enter debug mode.', action='store_true', default=False)
args = parser.parse_args()

class NormalInverseWishart(object):
    def __init__(self, mu, kappa, psi, nu):
        """Creates a normal-inverse-Wishart distribution with parameters mu, kappa, psi, nu."""
        self._mu = mu
        self._kappa = kappa
        self._psi = psi
        self._nu = nu
        self._invwishart = invwishart(df=nu, scale=psi)

    def pdf(self, x, sigma_sq):
        sigma = np.sqrt(sigma_sq)
        invwishart_factor = self._invwishart.pdf(sigma)
        normal_distn = norm(self._mu, sigma/kappa)
        normal_factor = normal_distn.pdf(x)
        return normal_factor * invwishart_factor

    def rvs(self, size=1): # for sampling
        sigmas = self._invwishart.rvs(size=size)
        mus = np.concatenate([norm(self._mu, s/self._kappa).rvs(size=1) for s in sigmas])
        return zip(mus, sigmas)

    def mode(self):
        invwishart_mode = self._psi/(self._nu + self._psi.shape[0] + 1)
        return self._mu, invwishart_mode

def getAxesLimitsFromDistns(true_distn, prior_modal_distn):
    true_mean, true_cov = true_distn.mean, true_distn.cov
    true_x_lims = np.array([true_mean[0] - 4*true_cov[0,0], true_mean[0] + 4*true_cov[0,0]])
    true_y_lims = np.array([true_mean[1] - 4*true_cov[1,1], true_mean[1] + 4*true_cov[1,1]])
    prior_mean, prior_cov = prior_modal_distn.mean, prior_modal_distn.cov
    prior_x_lims = np.array([prior_mean[0] - 4*prior_cov[0,0], prior_mean[0] + 4*prior_cov[0,0]])
    prior_y_lims = np.array([prior_mean[1] - 4*prior_cov[1,1], prior_mean[1] + 4*prior_cov[1,1]])
    x_lower_lim = np.min([true_x_lims[0], prior_x_lims[0]])
    x_upper_lim = np.max([true_x_lims[1], prior_x_lims[1]])
    y_lower_lim = np.min([true_y_lims[0], prior_y_lims[0]])
    y_upper_lim = np.max([true_y_lims[1], prior_y_lims[1]])
    x_space = np.linspace(x_lower_lim, x_upper_lim, 100)
    y_space = np.linspace(y_lower_lim, y_upper_lim, 100)
    x_grid, y_grid = np.meshgrid(x_space, y_space)
    return x_grid, y_grid

def getPriorDistn(prior_mu, prior_kappa, prior_psi, prior_nu):
    return NormalInverseWishart(prior_mu, prior_kappa, prior_psi, prior_nu)

def plotTrueDistn(true_distn, x_grid, y_grid):
    plot_space = np.empty(x_grid.shape + (2,))
    plot_space[:,:,0] = x_grid
    plot_space[:,:,1] = y_grid
    surf = ax.plot_surface(x_grid, y_grid, true_distn.pdf(plot_space), cmap='viridis', linewidth=0)

true_distn = multivariate_normal(args.true_mean, np.sqrt(args.true_variance))
prior_distn_mode = getPriorDistn(np.array(args.prior_mean), args.prior_kappa, np.array(args.prior_variance), args.prior_nu).mode()
prior_modal_distn = multivariate_normal(prior_distn_mode[0], prior_distn_mode[1])
x_grid, y_grid = getAxesLimitsFromDistns(true_distn, prior_modal_distn)
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
plotTrueDistn(true_distn, x_grid, y_grid)
plt.show(block=False)

# TODO: Add process args function to make matrices from command line arguments
#   Add second surface plot for plotting posterior
#   Add titles
