"""
For estimating the parameters of a multinomial distribution using a Dirichlet prior.
A lot of the plotting was taken from https://gist.github.com/tboggs/8778945

Useful lines to include for editing:
    execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
"""
import os
execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import datetime as dt
import pandas as pd
from scipy.stats import multinomial, dirichlet

# command line parameters
parser = argparse.ArgumentParser(description='Demonstrate Bayesian inference for the parameters of a multinomial distribution.')
parser.add_argument('-t', '--true_probs', help='The naive probability parameters of the true distribution.', type=float, nargs=3)
parser.add_argument('-p', '--prior_params', help='The parameters of the Dirichlet prior', type=float, default=[1,1,1], nargs=3)
parser.add_argument('-n', '--num_data_points', help='The number of data points to sample from true distn.', type=int, default=50)
parser.add_argument('-d', '--debug', help='Enter debug mode.', action='store_true', default=False)
args = parser.parse_args()

def getTriangularBarycentricMesh(subdiv=8, tol=1.e-10):
    '''Converts 2D Cartesian coordinates to barycentric.
    Arguments:
        `subdiv` (int): Number of recursive mesh subdivisions to create.
        `tol` (float): tolerance
    '''
    _corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(0.75)]])
    _triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])
    _midpoints = [(_corners[(i + 1) % 3] + _corners[(i + 2) % 3])/2.0 for i in range(3)]
    refiner = tri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    num_cart_points = trimesh.x.shape[0]
    carts = zip(trimesh.x, trimesh.y)
    bary_trimesh = np.zeros([num_cart_points, 3])
    for i in range(0, num_cart_points):
        cart = carts[i]
        s = [(_corners[j] - _midpoints[j]).dot(cart - _midpoints[j])/0.75 for j in range(3)]
        bary_trimesh[i] = np.clip(s, tol, 1.0 - tol)
    return bary_trimesh, trimesh

def drawPdfContours(dist, bary_trimesh, cart_trimesh, nlevels=50, **kwargs):
    '''Draws pdf contours over an equilateral triangle (2-simplex).
    Arguments:
        `dist`: A distribution instance with a `pdf` method.
        `bary_trimesh`: A barycentric coord triangular mesh
        `border` (bool): If True, the simplex border is drawn.
        `nlevels` (int): Number of contours to draw.
        kwargs: Keyword args passed on to `plt.triplot`.
    '''
    pvals = dist.pdf(bary_trimesh.T)
    plt.tricontourf(cart_trimesh, pvals, nlevels, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, np.sqrt(0.75))
    plt.axis('off')
    plt.title(r'$\alpha$ = (%.3f, %.3f, %.3f)' % tuple(dist.alpha))

def getDirichletMode(dirichlet_distn):
    alpha = dirichlet_distn.alpha[dirichlet_distn.alpha > 1]
    return (alpha - 1)/(alpha.sum(dtype=float) - alpha.size)

def getTrueDistn(true_probs):
    if true_probs == None:
        football_data = pd.read_csv(os.path.join(os.environ['HOME'], 'Bayesian_Inference/csv/EPL20172018.csv'))
        naive_probs = football_data['FTR'].value_counts(normalize=True)
        true_distn = multinomial(1, naive_probs)
    else:
        true_distn = multinomial(1, true_probs)
    return true_distn

def getPriorDistn(prior_params):
    return dirichlet(prior_params)

def getPosteriorDistn(prior_params, x_data):
    post_params = prior_params + x_data.sum(axis=0)
    return dirichlet(post_params)

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Constructing true distribution...')
    true_distribution = getTrueDistn(args.true_probs)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Sampling...')
    x_data = true_distribution.rvs(size=args.num_data_points)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Constructing prior distribution...')
    prior_distribution = getPriorDistn(args.prior_params)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plotting prior distribution...')
    bary_trimesh, cart_trimesh = getTriangularBarycentricMesh()
    fig = plt.figure()
    drawPdfContours(prior_distribution, bary_trimesh, cart_trimesh); plt.show(block=False)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Integrating data and plotting posterior distribution...')
    for i in range(0, x_data.shape[0]):
        posterior_distribution = getPosteriorDistn(args.prior_params, x_data[:i])
        mesh_pdf_values = posterior_distribution.pdf(bary_trimesh.T)
        plt.tricontourf(cart_trimesh, mesh_pdf_values, 50)
        plt.title(r'$\alpha$ = (%.3f, %.3f, %.3f)' % tuple(posterior_distribution.alpha))
        plt.pause(0.05)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'true multinomial params = ' + str(true_distribution.p))
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'MAP estimated multinomial params = ' + str(getDirichletMode(posterior_distribution)))

if not(args.debug):
    main()
