"""
For estimating the rate of a Poisson variable using Maximum a posteriori with a Gamma prior.
Useful lines to include for editing:
    execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
"""
import os
execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import poisson, gamma

parser = argparse.ArgumentParser(description='Demonstrate Bayesian inference for the rate of a Poisson distribution.')
parser.add_argument('-t', '--true_lambda', help='The rate of the Poisson distribution.', type=float, default=5.0)
parser.add_argument('-p', '--prior_params', help='The parameters of the Gamma distribution ($\alpha, \beta$).', type=float, nargs=2, default=[9.0, 2.0])
parser.add_argument('-n', '--num_data_points', help='The number of data points to generate and use.', type=int, default=100)
parser.add_argument('-d', '--debug', help='Enter debug mode.', action='store_true', default=False)
args = parser.parse_args()

def getTrueDistn(true_lambda):
    return poisson(true_lambda)

def getPriorDistn(alpha, beta):
    return gamma(alpha, scale=1/beta)

def getPosteriorDistn(alpha, beta, x_data):
    post_alpha = alpha + x_data.sum()
    post_beta = beta + x_data.size
    return gamma(post_alpha, scale=1/post_beta)

def getGammaMode(gamma_distn):
    alpha = gamma_distn.args[0]
    beta = 1/gamma_distn.kwds['scale']
    if alpha <= 1.0:
        mode = 0.0
    else:
        mode = (alpha - 1)/beta
    return mode

def plotPriorDistn(prior_probs, x_axis_points):
    plt.plot(x_axis_points, prior_probs, color='g')
    plt.fill_between(x_axis_points, prior_probs, color='green', alpha=0.3, label='prior pdf')
    plt.xlabel('$\lambda$'); plt.ylabel('$p(\lambda|\mathbf{x})$')

def plotPoissonDistn(poisson_probs, x_axis_points):
    plt.plot(x_axis_points, poisson_probs, color='g')
    plt.fill_between(x_axis_points, poisson_probs, color='green', alpha=0.3, label='prior pdf')
    plt.xlabel('$x$'); plt.ylabel('$p(x)$')

true_distn = getTrueDistn(args.true_lambda)
x_data = true_distn.rvs(size=args.num_data_points)
prior_distn = getPriorDistn(args.prior_params[0], args.prior_params[1])
x_axis_points = np.linspace(0, 10, 1000)
x_axis_discrete = range(0,11)
prior_probs = prior_distn.pdf(x_axis_points)
fig = plt.figure()
plt.subplot(211)
plotPriorDistn(prior_probs, x_axis_points)
prior_poisson_distn = poisson(getGammaMode(prior_distn))
poisson_probs = prior_poisson_distn.pmf(x_axis_discrete)
plt.subplot(212)
plotPoissonDistn(poisson_probs, x_axis_discrete)
