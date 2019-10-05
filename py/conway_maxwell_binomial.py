"""
For defining the Conway-Maxwell binomial distribution in a class, calculating some probabilities and taking some samples.
"""
import os, sys
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, logit
from math import log, factorial

parser = argparse.ArgumentParser(description='For defining the Conway-Maxwell binomial distribution in a class, calculating some probabilities and taking some samples.')
parser.add_argument('-p', '--success_prob', help='Probability of success.', type=float, default=0.5)
parser.add_argument('-n', '--nu_values', help='Values of nu to use.', nargs = '*', default=[1.0, 0.5, 1.5], type=float)
parser.add_argument('-m', '--number_of_bernoulli', help='Number of bernoulli variables to use.', default=50, type=int)
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
        self.normaliser = self.getNormaliser()
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
        if self.p == 1:
            p_k = 1 if k == self.m else 0
        elif self.p == 0:
            p_k = 1 if k == 0 else 0
        else:
            p_k = np.exp((self.nu * log(comb(self.m, k))) + (k*log(self.p)) + ((self.m-k) * log(1-self.p)))/self.normaliser
        return p_k
    
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

    def getNormaliser(self):
        """
        For calculating the normalising factor of the distribution.
        Arguments:  self, the distribution object
        Returns:    the value of the normalising factor S(p,nu)
        """
        if (self.p == 0) | (self.p == 1):
            warnings.warn("p = " + str(self.p) + " The distribution is deterministic.")
            return 0
        else:
            return np.sum([np.exp((self.nu * log(comb(self.m, i))) + (i * log(self.p)) + ((self.m - i) * log(1-self.p))) for i in range(0, self.m + 1)])

def getLogFactorial(k):
    """
    For calculating log(k!) = log(k) + log(k-1) + ... + log(2) + log(1).
    Arguments:  k, int
    Returns:    log(k!)
    """
    return np.sum([log(i) for i in range(1, k+1)])

def calculateSecondSufficientStat(samples,m):
    """
    For calculating the second sufficient stat for the conway maxwell binomial distribution. k!(m-k)!
    if samples is an array, sums over the samples.
    Arguments:  samples, integer or array of integers
                m, the maximum value of any given sample
    Returns:    \sum_{i=1}^n k_i! (m - k_i)! where k_i is a sample
    """
    samples = np.array([samples]) if np.isscalar(samples) else samples
    return np.sum([getLogFactorial(sample) + getLogFactorial(m - sample) for sample in samples])

def posteriorProprietyTest(a,b,c,m):
    """
    The conjugate posterior of the Conway-Maxwell binomial distribution is only proper for certain values of the hyperparameters.
    This function tests if the hyperparameters are within these values.
    Arguments:  a, float, the first hyperparameter, corresponds to the first sufficient statistic \sum k_i
                b, float, the second hyperparameter, corresponds to the seconds sufficient statistic, \sum log(k_i!(m-k_i)!)
                c, int, the pseudocount hyperparameter.
    Returns:    None, or raises an error
    """
    assert 0 < (a/c), "a/c <= 0"
    assert (a/c) < m, "a/c >= m"
    assert -getLogFactorial(m) < (b/c), "-log(m!) >= b/c"
    floored_ratio = np.floor(a/c).astype(int)
    ceiled_ratio = np.ceil(a/c).astype(int)
    larger = -calculateSecondSufficientStat(floored_ratio,m) + ((a/c) - floored_ratio)*(-calculateSecondSufficientStat(ceiled_ratio,m) + calculateSecondSufficientStat(floored_ratio,m))
    assert (b/c) < larger, "(b/c) > t(floor(a/c)) + (a/c - floor(a/c))(t(ceil(a/c)) - t(floor(a/c)))"
    return None

def conwayMaxwellBinomialPosteriorKernel(params, prior_params, suff_stats, m, n):
    """
    For calculating the kernel of the posterior distribution of a Conway-Maxwell binomial distribution at parameter values 'params'.
    Parameters are assumed to be in canonical form, rather than natural.
    Arguments:  params, 2 element 1-d numpy array (float), the parameter values for the Conway-Maxwell binomial distribution
                prior_params, 2 element list, first element is 2 element 1-d numpy array (float), second element is int, parameters of the prior distribution
                suff_stats, 2 element array, sufficient statistics of the Conway-Maxwell binomial distribution, calculated from data, (sum(k_i), sum(log(k_i!(m-k_i!))))
                m, int, the number of bernoulli variables, considered fixed and known.
                n, int, number of data points
    Returns: the kernel value at (p, nu) = params
    """
    p, nu = params
    chi, c = prior_params
    a, b = chi
    posteriorProprietyTest(a,b,c,m)
    natural_params = np.array([logit(p), nu])
    data_part = np.dot(natural_params, chi + suff_stats)
    total_count = c + n # includes psuedocounts
    test_dist = ConwayMaxwellBinomial(p, nu, m)
    partition_part = np.log(test_dist.normaliser) - (nu * getLogFactorial(m)) - (m * np.log(1-p))
    return np.exp(data_part - total_count*partition_part)

def calcUpperBound(a,c,m):
    ratio=a/c
    floored_ratio = np.floor(ratio).astype(int)
    ceiled_ratio = np.ceil(ratio).astype(int)
    return -calculateSecondSufficientStat(floored_ratio,m) + (ratio - floored_ratio)*(-calculateSecondSufficientStat(ceiled_ratio,m) + calculateSecondSufficientStat(floored_ratio,m))

def plotConwayMaxwellPmf(com_dist):
    """
    For plotting the pmf of the given distribution. 
    Arguments:  com_dist, distribution
    Returns:    None
    """
    plt.plot(range(0, com_dist.m + 1), list(com_dist.samp_des_dict.values()))
    plt.xlim(0,com_dist.m)
    plt.xlabel('k')
    plt.ylabel('P(k)')

bernoulli_dist = ConwayMaxwellBinomial(0.5, 1, 1)
binom_dist = ConwayMaxwellBinomial(0.5, 1, 50)
over_disp_dist = ConwayMaxwellBinomial(0.5, 0.5, 50)
under_disp_dist = ConwayMaxwellBinomial(0.5, 1.5, 50)

# need to sample from a distribution
n=50
samples = binom_dist.rvs(size=n)
# then define the prior parameters
m=50 # technically a parameter of the distributions, but considered known and fixed
prior_params = [np.array([m/2, -120]), 1]
possible_p_values = np.linspace(0,1, 101)
possible_nu_values = np.linspace(-1, 2, 301)

kernel_values = np.array([conwayMaxwellBinomialPosteriorKernel(np.array([p,1]), prior_params, np.array([samples.sum(), -calculateSecondSufficientStat(samples, m)]), m, n) for p in possible_p_values])
plt.subplot(1,2,1)
plt.plot(possible_p_values, kernel_values)
kernel_values = np.array([conwayMaxwellBinomialPosteriorKernel(np.array([0.5,nu]), prior_params, np.array([samples.sum(), -calculateSecondSufficientStat(samples, m)]), m, n) for nu in possible_nu_values])
plt.subplot(1,2,2)
plt.plot(possible_nu_values, kernel_values)
plt.show(block=False)
