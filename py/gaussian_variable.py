"""
For estimating the parameters of a Gaussian variable using Maximum a posteriori with a Gaussian prior.
Useful lines to include for editing:
    execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
"""
import os
execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import norm, invgamma
from math import gamma, pi

# command line arguments
parser = argparse.ArgumentParser(description='Demonstrate Bayesian inference for the parameters of a Gaussian distribution.')
parser.add_argument('-t', '--true_params', help='True mean and variance of the true distn.', type=float, nargs=2, default=[1.2, 0.3])
parser.add_argument('-p', '--prior_params', help='The parameters of the Normal Inverse Gaussian prior', type=float, default=[0.0, 0.1, 0.2, 0.3], nargs=4)
parser.add_argument('-n', '--num_data_points', help='The number of data points to sample from true distn.', type=int, default=100)
parser.add_argument('-d', '--debug', help='Enter debug mode.', action='store_true', default=False)
args = parser.parse_args()

class NormalInverseGamma(object):
    def __init__(self, mu, lamda, alpha, beta): # misspelling of lambda here is intentional
        """Creates a univariate NormalInverseGamma distribution with parameters mu, lambda, alpha, beta."""
        self._mu = mu
        self._lambda = lamda
        self._alpha = alpha
        self._beta = beta
        self._coef = np.sqrt(self._lambda/2*pi)*np.power(self._beta, self._alpha)/gamma(self._alpha)

    def pdf(self, x, sigma_sq): # probability density function
        sigma = np.sqrt(sigma_sq)
        exponential_factor = exp(-(2*self._beta + self._lambda*np.power((x - self._mu),2))/2*sigma_sq)
        sigma_factor = (1/sigma)*np.power(1/sigma_sq, self._alpha + 1)
        return self._coef * exponential_factor * sigma_factor

    def rvs(self, size=1): # for sampling
        sigma_sqs = invgamma(self._alpha, self._beta).rvs(size=size)
        mus = np.concatenate([norm(self._mu, s).rvs(size=1) for s in sigma_sqs])
        return zip(mus, sigma_sqs)

    def mode(self): # returns modal mean and variance
        sigma_sq = self._beta/(self._alpha + 1 + 0.5)
        return self._mu, sigma_sq

def getXlimsFromDistns(true_distn, prior_distn):
    true_lims = np.array([true_distn.args[0] - 3*np.sqrt(true_distn.args[1]), true_distn.args[0] + 3*np.sqrt(true_distn.args[1])])
    prior_mode_mu, prior_mode_sigma_sq = prior_distn.mode()
    prior_lims = np.array([prior_mode_mu - 3*np.sqrt(prior_mode_sigma_sq), prior_mode_mu + 3*np.sqrt(prior_mode_sigma_sq)])
    all_lims = np.concatenate([true_lims, prior_lims])
    return np.linspace(all_lims.min(), all_lims.max(), 100), prior_mode_mu, prior_mode_sigma_sq

def getPriorDistn(mu, lamda, alpha, beta):
    return NormalInverseGamma(mu, lamda, alpha, beta)

def getPosteriorModalDistn(prior_mu, prior_lambda, prior_alpha, prior_beta, x_data):
    x_n, x_mean = [x_data.size, x_data.mean()]
    post_mu = (prior_lambda*prior_mu + x_n*x_mean)/(prior_lambda + x_n)
    post_lambda = prior_lambda + x_n
    post_alpha = prior_alpha + x_n/2
    post_beta = prior_beta + np.power(x_data - x_mean, 2).sum()/2 + (x_n*prior_lambda*np.power((x_mean - prior_mu),2)/(2*(prior_lambda + x_n)))
    post_distn = NormalInverseGamma(post_mu, post_lambda, post_alpha, post_beta)
    post_mu, post_sigma_sq =  post_distn.mode()
    post_modal_distn = norm(post_mu, np.sqrt(post_sigma_sq))
    return post_modal_distn

def plotPriorDistn(prior_probs, values_for_plotting):
    plt.plot(values_for_plotting, prior_probs, color='g')
    plt.fill_between(values_for_plotting, prior_probs, color='green', alpha=0.3, label='Gaussian pdf at prior mode')
    plt.xlabel('$x$'); plt.ylabel('$f(x | \mu, \sigma^2)$')
    plt.tight_layout()

def plotPosteriorDistn(posterior_probs, values_for_plotting, num_data_points):
    if num_data_points == 3:
        plt.plot(values_for_plotting, posterior_probs, 'r', alpha=0.3, label='Gaussian pdf at posterior mode')
    else:
        plt.plot(values_for_plotting, posterior_probs, 'r', alpha=0.3)
    plt.title('Number of data points used: ' + str(num_data_points))

def plotTrueDistn(true_probs, values_for_plotting):
    plt.plot(values_for_plotting, true_probs, color='b')
    plt.fill_between(values_for_plotting, true_probs, color='blue', alpha=0.3, label='true Gaussian pdf')

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Constructing true distribution...')
    args.true_params[1] = np.sqrt(args.true_params[1])
    true_distn = norm(args.true_params[0], args.true_params[1])
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Sampling...')
    x_data = true_distn.rvs(size=args.num_data_points)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Constructing prior distribution...')
    prior_distn = getPriorDistn(args.prior_params[0], args.prior_params[1], args.prior_params[2], args.prior_params[3])
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plotting Gaussian pdf at prior mode...')
    values_for_plotting, prior_mode_mu, prior_mode_sigma_sq = getXlimsFromDistns(true_distn, prior_distn)
    prior_modal_distn = norm(prior_mode_mu, np.sqrt(prior_mode_sigma_sq))
    prior_probs = prior_modal_distn.pdf(values_for_plotting)
    fig = plt.figure()
    plotPriorDistn(prior_probs, values_for_plotting)
    plt.show(block=False)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Integrating data and plotting Gaussian pdf at posterior mode...')
    for i in range(2, x_data.shape[0]): # starting at 2 to avoid a 0 variance
        posterior_modal_distn = getPosteriorModalDistn(args.prior_params[0], args.prior_params[1], args.prior_params[2], args.prior_params[3], x_data[:i])
        posterior_probs = posterior_modal_distn.pdf(values_for_plotting)
        plotPosteriorDistn(posterior_probs, values_for_plotting, i+1)
        plt.pause(0.05)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plotting true pdf...')
    true_probs = true_distn.pdf(values_for_plotting)
    plotTrueDistn(true_probs, values_for_plotting)
    plt.legend()
    plt.show(block=False)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'true parameters = ' + str(args.true_params))
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'MAP estimated parameters = ' + str(posterior_modal_distn.args))

if not(args.debug):
    main()
