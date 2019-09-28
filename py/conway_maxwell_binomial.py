"""
For defining the Conway-Maxwell binomial distribution in a class, calculating some probabilities and taking some samples.
"""
import os, sys
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from math import log, factorial

parser = argparse.ArgumentParser(description='For defining the Conway-Maxwell binomial distribution in a class, calculating some probabilities and taking some samples.')
parser.add_argument('-p', '--success_prob', help='Probability of success.', type=float, default=0.5)
parser.add_argument('-n', '--nu_values', help='Values of nu to use.', nargs = '*', default=[1.0, 0.5, 1.5], type=float)
parser.add_argument('-m', '--number_of_trials', help='Number of trials to use.', default=50, type=int)
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

class ConwayMaxwellBinomial(object):
    def __init__(self, p, nu, m):
        """
        Creates the Conway-Maxwell binomial distribution with parameters p, nu, and m. Calculates the normalising function during initialisation. Uses exponents and logs to avoid overflow.
        Arguments:  self,
                    p, real 0 <= p <= 1, probability of success
                    nu, real, dispersion parameter
                    m, number of trials
        Returns:    object
        """
        self.p = p
        self.nu = nu
        self.m = m
        self.normaliser = np.sum([np.exp((self.nu * log(comb(self.m, i))) + (i * log(self.p)) + ((self.m - i) * log(1-self.p))) for i in range(0, self.m + 1)])
        self.samp_des_dict = self.getSamplingDesignDict()

    def pmf(self, k):
        """
        Probability mass function. Uses exponents and logs to avoid overflow.
        Arguments:  self, ConwayMaxwellBinomial object,
                    k, int, must be an integer in the interval [0, m]
        Returns:    P(k)
        """
        if (k > self.m) | (k != int(k)):
            raise ValueError("k must be an integer between 0 and m, inclusive")
        return np.exp((self.nu * log(comb(self.m, k))) + (k*log(self.p)) + ((self.m-k) * log(1-self.p)))/self.normaliser
    
    def getSamplingDesignDict(self):
        """
        Returns a dictionary representing the sampling design of the distribution. That is, samp_des_dict[k] = pmf(k)
        Arguments:  self, the distribution object,
        Returns:    samp_des_dict, dictionary, int => float
        """
        samp_des_dict = {k:self.pmf(k)for k in range(0,self.m + 1)}
        return samp_des_dict

    def rvs(self, size=1):
        return np.random.choice(range(0,self.m + 1), size=size, replace=True, p=list(self.samp_des_dict.values()))

class ConwayMaxwellBinomialConjugatePrior(object):
    def init(self, chi, c, m):
        """
        For initialising a conjugate prior for the Conway-Maxwell binomial distribution. 
        Arguments:  self, object.
                    chi, 2-d hyperparameters.
                    c, the count hyperparameter.
                    m, the number of trials in the Conway-Maxwell distribution
        Returns:    object
        """
        self.chi = chi
        self.c = c
        self.m = m
        self.con_normaliser = np.sum([(comb(self.m, i)**self.chi[1]) * np.exp(i*self.chi[0]) / (1 + np.exp(self.chi[0]))**m  for i in range(0, self.m + 1)])

    def pdf(self, x):
        """
        Probability Density fuction. Returns a quantity proportional to p(x).
        Arguments:  self, object.
                    x, 2-d array. 
        Returns:    P(x)
        """
        return np.dot(self.chi, x) - (c * log(factorial(self.m))) + (log(self.con_normaliser))

def paramsToNatural(params):
    """
    Transforms the given parameters for the conway-maxwell binomial distribution to the natural parameters.
    Arguments:  params, 2 element 1-d array, p, nu.
    Returns:    2 element 1-d array, log(p/(1-p)), nu
    """
    p, nu = params
    return log(p/1-p), nu

def naturalToParams(natural):
    """
    Transforms the given natural parameters of the Conway-maxwell binomial distribution into the intuitive parameters.
    Arguments:  natural, 2 element 1-d array, eta, nu
    Returns:    2 element 1-d array, 1/(1 + exp(-eta)), nu
    """
    eta, nu = natural
    return 1/(1 + np.exp(-eta)), nu

bernoulli_dist = ConwayMaxwellBinomial(0.5, 1, 1)
binom_dist = ConwayMaxwellBinomial(0.5, 1, 50)
over_disp_dist = ConwayMaxwellBinomial(0.5, 0.5, 50)
under_disp_dist = ConwayMaxwellBinomial(0.5, 1.5, 50)

# need to sample from a distribution
m=100
samples = bernoulli_dist.rvs(size=m)
# then define the prior parameters
prior_params = [paramsToNatural(0.5, 0),1] 
# then define the posterior distribution
posterior_dist = ConwayMaxwellBinomialConjugatePrior([prior_params[0] + samples.mean(), ])
# then maximise the pdf of the posterior
# the parameters at the mode are what I'm looking for

for nu in args.nu_values:
    com_bin_dist = ConwayMaxwellBinomial(args.success_prob, nu, args.number_of_trials)
    plt.plot(range(0,args.number_of_trials + 1), list(com_bin_dist.samp_des_dict.values()), label='nu = ' + str(nu))
plt.legend(fontsize='large')
plt.xlabel('k', fontsize='large')
plt.ylabel('P(k)', fontsize='large')
plt.title('p = ' + str(args.success_prob) + ', num trials = ' + str(args.number_of_trials), fontsize='large')
plt.tight_layout()
plt.show(block=False)

