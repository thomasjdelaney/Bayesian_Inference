"""
For estimating the mean value of a Bernoulli variable using Maximum a posteriori with a Beta prior.
Useful lines to include for editing:
    execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import beta, binom

# command line arguments
parser = argparse.ArgumentParser(description='Demonstrate Bayesian inference.')
parser.add_argument('-t', '--true_mu', help='True mean of Bernoulli distn.', type=float, default=0.2)
parser.add_argument('-p', '--prior_params', help='The parameters of the beta prior', type=float, default=[10,10], nargs=2)
parser.add_argument('-n', '--num_data_points', help='The number of data points to sample from true distn.', type=int, default=100)
parser.add_argument('-d', '--debug', help='Enter debug mode.', action='store_true', default=False)
args = parser.parse_args()

def getBetaMode(beta_distn):
    a, b = beta_distn.args
    if (a == 1) & (b == 1):
        print(dt.datetime.now().isoformat() + ' WARN: ' + 'Distribution is uniform along (0,1)!')
        mode = np.random.rand()
    elif (a == 1) & (b > 1):
        mode = 0.0
    elif (a > 1) & (b == 1):
        mode = 1.0
    else:
        mode = (a - 1.0)/(a + b - 2.0)
    return mode

def getTrueDistn(mu):
    return binom(1, mu)

def getPriorDistn(a,b):
    return beta(a,b)

def getPosteriorDistn(a,b,x_data):
    a_n = a + x_data.sum()
    b_n = b + (x_data.shape[0] - x_data.sum())
    return beta(a_n, b_n)

def plotPriorDistn(prior_probabilities, possible_mu_values):
    plt.plot(possible_mu_values, prior_probabilities, color='g')
    plt.fill_between(possible_mu_values, prior_probabilities, color='green', alpha=0.3, label='prior pdf')
    plt.xlabel('$\mu$'); plt.ylabel('$p(\mu|\mathbf{x})$')
    plt.tight_layout()

def plotPosteriorDistn(posterior_probabilities, possible_mu_values, num_data_points):
    if num_data_points == 1:
        plt.plot(possible_mu_values, posterior_probabilities, 'r', alpha=0.3, label='posterior pdf')
    else:
        plt.plot(possible_mu_values, posterior_probabilities, 'r', alpha=0.3)
    plt.title('Number of data points used: ' + str(num_data_points))

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Constructing true distribution...')
    true_distribution = getTrueDistn(args.true_mu)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Sampling...')
    x_data = true_distribution.rvs(size=args.num_data_points)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Constructing prior distribution...')
    prior_distribution = getPriorDistn(args.prior_params[0], args.prior_params[1])
    possible_mu_values = np.linspace(0, 1, 100)
    prior_probabilities = prior_distribution.pdf(possible_mu_values)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plotting prior distribution...')
    fig = plt.figure()
    plotPriorDistn(prior_probabilities, possible_mu_values)
    plt.show(block=False)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Integrating data and plotting posterior distribution...')
    for i in range(0, x_data.shape[0]):
        posterior_distn = getPosteriorDistn(args.prior_params[0], args.prior_params[1], x_data[:i])
        posterior_probabilities = posterior_distn.pdf(possible_mu_values)
        plotPosteriorDistn(posterior_probabilities, possible_mu_values, i+1)
        plt.pause(0.05)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plotting true mu...')
    plt.vlines(args.true_mu, ymin=0, ymax=plt.ylim()[1], label='true $\mu$', alpha=0.3, linestyles='dashed')
    plt.legend()
    plt.show(block=False)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'true mu = ' + str(args.true_mu))
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'MAP estimated mu = ' + str(getBetaMode(posterior_distn)))

if not(args.debug):
    main()
