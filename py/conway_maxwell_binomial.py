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
from scipy.special import comb
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

def calculateSecondSufficientStat(samples,m):
  """
  For calculating the second sufficient stat for the conway maxwell binomial distribution. k!(m-k)!
  if samples is an array, sums over the samples.
  Arguments:  samples, integer or array of integers
              m, the maximum value of any given sample
  Returns:    \sum_{i=1}^n k_i! (m - k_i)! where k_i is a sample
  """
  samples = np.array([samples]) if np.isscalar(samples) else samples
  return np.sum([log(factorial(sample))  +  log(factorial(m - sample)) for sample in samples])

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
  test_dist = ConwayMaxwellBinomial(p, nu, m)
  natural_params = np.array([np.log(p/(1-p)), -nu])
  total_count = c + n # includes psuedocounts
  partition_part = np.log(test_dist.normaliser) - (m*np.log(1-p)) - (nu*np.log(factorial(m)))
  data_part = np.dot(natural_params, chi + suff_stats)
  return np.exp(data_part - (total_count*partition_part))

bernoulli_dist = ConwayMaxwellBinomial(0.5, 1, 1)
binom_dist = ConwayMaxwellBinomial(0.5, 1, 50)
over_disp_dist = ConwayMaxwellBinomial(0.5, 0.5, 50)
under_disp_dist = ConwayMaxwellBinomial(0.5, 1.5, 50)

# need to sample from a distribution
n=100
samples = binom_dist.rvs(size=n)
# then define the prior parameters
m=50 # technically a parameter of the distributions, but considered known and fixed
prior_params = [np.array([1, 0.001]),1]
possible_p_values = np.linspace(0,1, 101)
possible_nu_values = np.linspace(-1, 1, 201)

kernel_values = np.array([conwayMaxwellBinomialPosteriorKernel(np.array([p,1]), prior_params, np.array([samples.sum(), calculateSecondSufficientStat(samples, m)]), m, n) for p in possible_p_values])
plt.subplot(1,2,1)
plt.plot(possible_p_values, kernel_values)
kernel_values = np.array([conwayMaxwellBinomialPosteriorKernel(np.array([0.5,nu]), prior_params, np.array([samples.sum(), calculateSecondSufficientStat(samples, m)]), m, n) for nu in possible_nu_values])
plt.subplot(1,2,2)
plt.plot(possible_nu_values, kernel_values)

#for nu in args.nu_values:
#    com_bin_dist = ConwayMaxwellBinomial(args.success_prob, nu, args.number_of_trials)
#    plt.plot(range(0,args.number_of_trials + 1), list(com_bin_dist.samp_des_dict.values()), label='nu = ' + str(nu))
#plt.legend(fontsize='large')
#plt.xlabel('k', fontsize='large')
#plt.ylabel('P(k)', fontsize='large')
#plt.title('p = ' + str(args.success_prob) + ', num trials = ' + str(args.number_of_trials), fontsize='large')
#plt.tight_layout()
#plt.show(block=False)

