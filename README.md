A repository for demonstrating some toy examples of Bayesian inference.

Python 2.7.15

Required packages:
1. argparse 1.1
2. numpy 1.15.2
3. matplotlib 2.2.3
4. datetime
5. scipy 1.1.0
6. pandas 0.23.4
7. math


For the default examples run:

* python -i py/bernoulli_variable.py
* python -i py/multinomial_variable.py
* python -i py/gaussian_variable.py


TODO: Prevent Gamma function overflow in gaussian_variable.py, extend multinomial_variable to more than three categories, prevent warning in poisson script, get suitable x axis range from true lambda in poisson script.
