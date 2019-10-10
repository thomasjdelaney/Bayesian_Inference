"""
For defining the Conway-Maxwell binomial distribution in a class, calculating some probabilities and taking some samples.
"""
import os, sys
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import argparse
import warnings
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, logit
from math import log, factorial
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from scipy.stats import norm
from scipy.optimize import minimize

parser = argparse.ArgumentParser(description='For defining the Conway-Maxwell binomial distribution in a class, calculating some probabilities and taking some samples.')
parser.add_argument('-p', '--params', help='Parameters for the Conway-Maxwell binomial distribution.', type=float, nargs=2, default=[0.5,1.0])
parser.add_argument('-r', '--init', help='Initial guess for the paramters when fitting.', type=float, nargs=2, default=[0.05, -0.05])
parser.add_argument('-m', '--num_bernoulli', help='Number of bernoulli variables to use.', default=50, type=int)
parser.add_argument('-n', '--num_samples', help='The number of samples to take.', default=50, type=int)
parser.add_argument('-s', '--save_fig', help='Flag to save the figure instead of showing it.', default=False, action='store_true')
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

proj_dir = os.path.join(os.environ['SPACE'], 'Bayesian_Inference')
image_dir = os.path.join(proj_dir, 'images')

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

def getSecondHyperparam(m):
    """
    Return a value for b that will centre the prior on nu=1.
    """
    numerator = -np.sum([calculateSecondSufficientStat(k,m) * comb(m,k) for k in range(0,m+1)])
    return numerator/(2**m)

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

def calcUpperBound(a,c,m):
    """
    Helper function for conjugateProprietyTest.
    """
    ratio=a/c
    floored_ratio = np.floor(ratio).astype(int)
    ceiled_ratio = np.ceil(ratio).astype(int)
    return -calculateSecondSufficientStat(floored_ratio,m) + (ratio - floored_ratio)*(-calculateSecondSufficientStat(ceiled_ratio,m) + calculateSecondSufficientStat(floored_ratio,m))

def conjugateProprietyTest(a,b,c,m):
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
    assert (b/c) < calcUpperBound(a,c,m), "(b/c) > t(floor(a/c)) + (a/c - floor(a/c))(t(ceil(a/c)) - t(floor(a/c)))"
    return None

def conwayMaxwellBinomialPriorKernel(com_params, a, b, c, m):
    """
    For calculating the kernel of the conjugate prior of the Conway-Maxwell binomial distribution. 
    Arguments:  com_params, p, nu, the parameters of the Conway-Maxwell binomial distribution
                a, hyperparameter corresponding to the first sufficient stat,
                b, hyperparameter corresponding to the second sufficient stat,
                c, hyperparameter corresponding to the pseudocount
                m, int, the number of bernoulli variables, considered fixed and known
    Returns:    The value of the kernel of the conjugate prior 
    """
    conjugateProprietyTest(a,b,c,m)
    # propriety_dist = norm(0, 1)
    p, nu = com_params
    if (p == 1) | (p == 0):
        return 0
    test_dist = ConwayMaxwellBinomial(p, nu, m)
    natural_params = np.array([logit(p), nu])
    pseudodata_part = np.dot(natural_params, np.array([a,b]))
    partition_part = np.log(test_dist.normaliser) - (nu * getLogFactorial(m)) - (m * np.log(1-p))
    # propriety_part = norm.pdf(logit(p)) * norm.pdf(nu - 1)
    return np.exp(pseudodata_part - c * partition_part)

def conwayMaxwellBinomialPosteriorKernel(com_params, a, b, c, suff_stats, m, n):
    """
    For calculating the kernel of the posterior distribution of a Conway-Maxwell binomial distribution at parameter values 'params'.
    Parameters are assumed to be in canonical form, rather than natural.
    Arguments:  com_params, 2 element 1-d numpy array (float), the parameter values for the Conway-Maxwell binomial distribution
                a, hyperparameter corresponding to the first sufficient stat,
                b, hyperparameter corresponding to the second sufficient stat,
                c, hyperparameter corresponding to the pseudocount
                suff_stats, 2 element array, sufficient statistics of the Conway-Maxwell binomial distribution, calculated from data, (sum(k_i), sum(log(k_i!(m-k_i!))))
                m, int, the number of bernoulli variables, considered fixed and known.
                n, int, number of data points
    Returns: the kernel value at (p, nu) = params
    """
    # propriety_dist = norm(0, 1)
    conjugateProprietyTest(a,b,c,m)
    p, nu = com_params
    if (p == 1) | (p == 0):
        return 0
    chi = np.array([a, b])
    natural_params = np.array([logit(p), nu])
    data_part = np.dot(natural_params, chi + suff_stats)
    test_dist = ConwayMaxwellBinomial(p, nu, m)
    partition_part = np.log(test_dist.normaliser) - (nu * getLogFactorial(m)) - (m * np.log(1-p))
    # propriety_part = norm.pdf(logit(p)) * norm.pdf(nu - 1)
    total_count = n + c # includes pseudocounts
    return np.exp(data_part - total_count * partition_part)

def conwayMaxwellNegLogLike(params, m, samples):
    """
    For calculating the negative log likelihood at p,nu.
    Arguments:  params: p, 0 <= p <= 1
                        nu, float, dispersion parameter
                m, number of bernoulli variables
                samples, ints between 0 and m, data.
    Returns:    float, negative log likelihood
    """
    p, nu = params
    if (p == 1) | (p == 0):
        return np.infty
    n = samples.size
    com_dist = ConwayMaxwellBinomial(p, nu, m)
    p_part = np.log(p/(1-p))*samples.sum()
    nu_part = nu * calculateSecondSufficientStat(samples,m)
    partition_part = np.log(com_dist.normaliser) - (nu * getLogFactorial(m)) - (m * np.log(1-p))
    return -(p_part - nu_part - n * partition_part)

def estimateParams(m, samples, init):
    """
    For estimating the parameters of the Conway-Maxwell binomial distribution from the given samples.
    Arguments:  m, the number of bernoulli variables being used.
                samples, ints, between 0 and m
    Return:     the fitted params, p and nu
    """
    bnds = ((0.000001,0.99999999),(-2,2))
    res = minimize(conwayMaxwellNegLogLike, init, args=(m,samples), bounds=bnds)
    return res.x

def plotConwayMaxwellPmf(com_dist, title='', **kwargs):
    """
    For plotting the pmf of the given distribution. 
    Arguments:  com_dist, distribution
    Returns:    None
    """
    plt.plot(range(0, com_dist.m + 1), list(com_dist.samp_des_dict.values()), **kwargs)
    plt.xlim(0,com_dist.m)
    plt.xlabel('k', fontsize='large')
    plt.ylabel('P(k)', fontsize='large')
    plt.legend(fontsize='large')
    plt.title(title, fontsize='large') if title != '' else None

def plotSamples(samples, m, title=''):
    """
    For plotting a histogram of the samples taken from the true distribution.
    """
    plt.hist(samples, bins=range(0,m+1), align='left')
    plt.xlabel('k', fontsize='large'); plt.ylabel('Num Occurances', fontsize='large')
    plt.xlim(-0.5,m+0.5)
    plt.title(title, fontsize='large') if title != '' else None

def plotPriorDistribution(a, b, c, possible_nu_values, m):
    """
    For plotting the prior distribution. Must be a 3-d grid plot.
    """
    possible_p_values = np.linspace(0,1, 51)
    grid_p, grid_nu = np.meshgrid(possible_p_values, possible_nu_values)
    prior_values = np.zeros(grid_p.shape)
    for i,j in product(range(grid_p.shape[0]), range(grid_p.shape[1])):
        prior_values[i,j] = conwayMaxwellBinomialPriorKernel([grid_p[i,j], grid_nu[i,j]], a, b, c, m)
    plt.contourf(grid_p, grid_nu, prior_values, levels=50)
    plt.xlabel('p', fontsize='large')
    plt.ylabel(r'$\nu$', fontsize='large')

def plotPosteriorDistribution(a, b, c, possible_nu_values, m, samples):
    """
    For plotting the posterior distribution. Must be a 3-d plot 
    """
    possible_p_values = np.linspace(0,1, 51)
    n = samples.size
    suff_stats = np.array([samples.sum(), -calculateSecondSufficientStat(samples, m)])
    grid_p, grid_nu = np.meshgrid(possible_p_values, possible_nu_values)
    posterior_values = np.zeros(grid_p.shape)
    for i,j in product(range(grid_p.shape[0]), range(grid_p.shape[1])):
        posterior_values[i,j] = conwayMaxwellBinomialPosteriorKernel([grid_p[i,j], grid_nu[i,j]], a, b, c, suff_stats, m, n)
    plt.contourf(grid_p, grid_nu, posterior_values, levels=50)
    plt.xlabel('p', fontsize='large')
    plt.ylabel(r'$\nu$', fontsize='large')

def plotLikeSurface(possible_nu_values, m, samples):
    """
    For plotting the likelihood surface
    """
    possible_p_values = np.linspace(0,1, 51)
    grid_p, grid_nu = np.meshgrid(possible_p_values, possible_nu_values)
    likelihood_vals = np.zeros(grid_p.shape)
    for i,j in product(range(grid_p.shape[0]), range(grid_p.shape[1])):
        likelihood_vals[i,j] = np.exp(-conwayMaxwellNegLogLike([grid_p[i,j], grid_nu[i,j]], m, samples))
    plt.contourf(grid_p, grid_nu, likelihood_vals, levels=50)
    plt.xlabel('p', fontsize='large')
    plt.ylabel(r'$\nu$', fontsize='large')
    plt.title('Likelihood surface', fontsize='large')

if (not args.debug) & (__name__ == "__main__"):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    [p, nu], m = args.params, args.num_bernoulli
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Creating true distribution...')
    true_distr = ConwayMaxwellBinomial(p, nu, m)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Sampling...')
    samples = true_distr.rvs(size=args.num_samples)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Fitting parameters...')
    est_p, est_nu = estimateParams(m, samples, args.init)
    fitted_dist = ConwayMaxwellBinomial(est_p, est_nu, m)
    possible_nu_values = np.linspace(nu-1, nu+1, 101)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plotting...')
    plt.figure(figsize=(14,5))
    plt.subplot(1,3,1)
    plotConwayMaxwellPmf(true_distr, title='Probability Mass Functions', label='True: p=' + str(p) + ', nu=' + str(nu))
    plotConwayMaxwellPmf(fitted_dist, label='Fitted: p=' + str(np.round(est_p,1)) + ', nu=' + str(np.round(est_nu,1)))
    plt.subplot(1,3,2)
    plotSamples(samples, m, 'Samples')
    plt.subplot(1,3,3)
    plotLikeSurface(possible_nu_values, m, samples)
    plt.tight_layout()
    file_name = os.path.join(image_dir, 'conway_maxwell_binomial_figures.png')
    plt.savefig(file_name) if args.save_fig else plt.show(block=False)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')
