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

parser = argparse.ArgumentParser(description='For defining the Conway-Maxwell binomial distribution in a class, calculating some probabilities and taking some samples.')
parser.add_argument('-p', '--success_prob', help='Probability of success.', type=float, default=0.5)
parser.add_argument('-n', '--nu_values', help='Values of nu to use.', nargs = '*', default=[1.0, 0.5, 1.5], type=float)
parser.add_argument('-m', '--number_of_trials', help='Number of trials to use.', default=50, type=int)
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

class ConwayMaxwellBinomial(object):
    def __init__(self, p, nu, m):
        """
        Creates the Conway-Maxwell binomial distribution with parameters p, nu, and m.
        Arguments:  self,
                    p, real 0 <= p <= 1, probability of success
                    nu, real, dispersion parameter
                    m, number of trials
        Returns:    object
        """
        self.p = p
        self.nu = nu
        self.m = m
        self.normaliser = np.sum([(comb(self.m, i)**self.nu) * (self.p**i) * ((1-self.p)**(self.m - i)) for i in range(0, self.m + 1)])
        self.samp_des_dict = self.getSamplingDesignDict()

    def pmf(self, k):
        """
        Probability mass function.
        Arguments:  self, ConwayMaxwellBinomial object,
                    k, int, must be an integer in the interval [0, m]
        Returns:    P(k)
        """
        if (k > self.m) | (k != int(k)):
            raise ValueError("k must be an integer between 0 and m, inclusive")
        return ((comb(self.m, k)**self.nu) * (self.p**k) * ((1-self.p)**(self.m-k)))/self.normaliser
    
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

bernoulli_dist = ConwayMaxwellBinomial(0.5, 1, 1)
binom_dist = ConwayMaxwellBinomial(0.5, 1, 50)
over_disp_dist = ConwayMaxwellBinomial(0.5, 0.5, 50)
under_disp_dist = ConwayMaxwellBinomial(0.5, 1.5, 50)

for nu in args.nu_values:
    com_bin_dist = ConwayMaxwellBinomial(args.success_prob, nu, args.number_of_trials)
    plt.plot(range(0,args.number_of_trials + 1), list(com_bin_dist.samp_des_dict.values()), label='nu = ' + str(nu))
plt.legend(fontsize='large')
plt.xlabel('k', fontsize='large')
plt.ylabel('P(k)', fontsize='large')
plt.tight_layout()
plt.show(block=False)

