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
