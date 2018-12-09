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
parser.add_argument('-p', '--prior_params', help='The parameters of the Gamma distribution ($\alpha, \beta$).', type=float, nargs=2, default=[2.0, 0.5])
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

def plotPriorDistn(prior_distn, x_axis_points):
    prior_probs = prior_distn.pdf(x_axis_points)
    plt.plot(x_axis_points, prior_probs, color='g')
    plt.fill_between(x_axis_points, prior_probs, color='green', alpha=0.3, label='prior pdf')
    plt.xlabel('$\lambda$'); plt.ylabel('$p(\lambda|\mathbf{x})$')

def plotPoissonDistn(poisson_distn, x_axis_points, colour='blue', **kwargs):
    poisson_probs = poisson_distn.pmf(x_axis_points)
    plt.step(x_axis_points, poisson_probs, color=colour, where='mid', **kwargs)
    plt.xlabel('$x$'); plt.ylabel('$p(x)$')

def plotPosteriorDistn(posterior_distn, x_axis_points, num_data_points, **kwargs):
    posterior_probs = posterior_distn.pdf(x_axis_points)
    plt.plot(x_axis_points, posterior_probs, 'r', alpha=0.3, **kwargs)
    plt.title('Number of data points used: ' + str(num_data_points))

def iteratePlots(gamma_alpha, gamma_beta, x_data_points, x_axis_points, x_axis_discrete):
    num_data_points = x_data_points.size
    posterior_distn = getPosteriorDistn(gamma_alpha, gamma_beta, x_data_points)
    plt.subplot(211)
    if num_data_points == 1:
        plotPosteriorDistn(posterior_distn, x_axis_points, num_data_points, label='posterior pdf')
    else:
        plotPosteriorDistn(posterior_distn, x_axis_points, num_data_points)
    post_poisson_distn = poisson(getGammaMode(posterior_distn))
    plt.subplot(212)
    plt.cla()
    if num_data_points == args.num_data_points-1:
        plotPoissonDistn(post_poisson_distn, x_axis_discrete, label='MAP estimated pmf')
    else:
        plotPoissonDistn(post_poisson_distn, x_axis_discrete)
    plt.pause(0.05)
    return posterior_distn

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Constructing true distribution...')
    true_distn = getTrueDistn(args.true_lambda)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Sampling...')
    x_data = true_distn.rvs(size=args.num_data_points)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Constructing prior distribution...')
    prior_distn = getPriorDistn(args.prior_params[0], args.prior_params[1])
    x_axis_points = np.linspace(0, 10, 1000)
    x_axis_discrete = range(0,11)
    fig = plt.figure(); plt.tight_layout();
    plt.subplot(211)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plotting prior distribution...')
    plotPriorDistn(prior_distn, x_axis_points)
    prior_poisson_distn = poisson(getGammaMode(prior_distn))
    poisson_probs = prior_poisson_distn.pmf(x_axis_discrete)
    plt.subplot(212)
    plotPoissonDistn(prior_poisson_distn, x_axis_discrete)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Integrating data and plotting posterior distribution...')
    iteratePlots(args.prior_params[0], args.prior_params[1], x_data[:0], x_axis_points, x_axis_discrete)
    for i in range(1, x_data.size):
    	posterior_distn = iteratePlots(args.prior_params[0], args.prior_params[1], x_data[:i], x_axis_points, x_axis_discrete)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plotting true distribution...')
    plotPoissonDistn(true_distn, x_axis_discrete, colour='orange', label='True pmf')
    plt.tight_layout();plt.legend(); plt.subplot(211); plt.legend();
    plt.show(block=False)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'true lambda = ' + str(args.true_lambda))
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'MAP estimated lambda = ' + str(getGammaMode(posterior_distn)))

if not(args.debug):
    main()
