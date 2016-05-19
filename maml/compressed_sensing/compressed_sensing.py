__author__ = 'epyzerknapp'

import numpy as np
import cPickle as pkl
from maml.compressed_sensing import twist_solver as twist
from copy import deepcopy
from maml.distances.tanimoto import tanimoto_similarity
"""
With compressed sensing we are looking to locate the maximum compression which we can use to reduce the dimensionanlity
for the feature representation. For each set, we calculate
"""


def compress(X, desired_feature_size, A=None, saveA=True, Afilename='compression_matrix.pkl'):
    """
    Compress a matrix of inputs using compressed sensing
    :param X: inputs, numpy array
    :param desired_feature_size: size of compressed features, int
    :param A: transformation matrix, numpy array, default=None
    :param saveA: dump the transformation matrix to a pickle, default=True
    :param Afilename: name of the pickle which is dumped if saveA is True
    :return: matrix of compressed inputs, numpy array
    """
    m, n = X.shape
    if not A:
        A = np.random.rand(desired_feature_size, m)
    if saveA:
        pkl.dump(A, Afilename)
    b = np.dot(A, X)
    return A, b.T


def solve(b, A, tolA=1e-7, tolD=1e-9, verbose=False, **kwargs):
    """
    Solves the x = A^-1 x b problem to reconstruct the signal/inputs
    :param b: compressed inputs, numpy array
    :param A:transformation matrix, numpy array
    :param tolA: tolerance for Twist in A
    :param tolD: tolerance for Twist in D
    :param verbose: print level
    :param kwargs: contains options for the TwIST minimizer
    :return: dictorary of results with keys:
        'lambdas', 'lambdas_debias', 'objective', 'times', 'debias_start', 'max_svd'
    """
    solver = twist.TwistSolver(tolA=tolA, tolD=tolD, verbose=verbose, **kwargs)
    values = solver.solve(b, A)
    labels = 'lambdas', 'lambdas_debias', 'objective', 'times', 'debias_start', 'max_svd'
    results = dict((label, value) for label, value in zip(labels, values))
    return results


def calc_error(X, lambdas):
    """
    Calculates reconstruction error for a compression
    :param X: uncompressed inputs
    :param A: compression matrix
    :param lambdas: expansion coefficients, from the TwIST solver
    :return: error dictionary
    """
    reconstructed_inputs = lambdas
    reconstructed_inputs[reconstructed_inputs>0.5] = 1
    reconstructed_inputs[reconstructed_inputs<0.5] = 0
    errors = dict()
    errors['l2_norm'] = np.linalg.norm(X - reconstructed_inputs, ord=2)/ np.linalg.norm(X, ord=2)
    errors['l_inf'] = np.linalg.norm(reconstructed_inputs - X, ord=np.Inf)
    errors['frobenius'] = np.linalg.norm(reconstructed_inputs - X, ord=None)
    errors['tanimoto'] = 1 - tanimoto_similarity(X, reconstructed_inputs)
    return errors


def optimize_compression(X, feature_size_min=10, feature_size_max=200, reconstruction_tolerance=1e-02,
                         error_convergence=None, error_type='l2_norm', epsilon=1):
    """
    This routine finds the best compression size which satisfies either a given reconstruction error, or a convergence
    in the reconstruction error wrt compression size.

    :param X: Uncompressed features
    :param feature_size_min: minimum size of compressed feature (i.e maximum compression factor)
    :param feature_size_max: maximum size of compressed feature (i.e. minimum compression factor)
    :param reconstruction_tolerance: the minimium reconstruction error to terminate the search
    :param error_convergence: a convergence in the error which would terminate the search, None for ignore
    :param error_type: the method for calculating the error can be

    * 'l2_norm' : Matrix -> 2-norm (largest sing. value) Vector -> sum(abs(x)**2)**(1./2)
    * 'l_inf' : Matrix -> max(sum(abs(x), axis=1)) Vector -> max(abs(x))
    * 'frobenius' : Matrix -> ||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2} Vector -> 2-norm

    :return: optimized A, reconstruction error
    """
    dim = feature_size_min
    a, b = compress(X, dim, saveA=False)
    converged = False
    res = solve(b, a, verbose=False, weight=1e-03, miniter=100, maxiter=2000)
    try:
        error = calc_error(X, res['lambdas'])
        old_error = error[error_type]
    except KeyError:
        raise StandardError('Error type {} not recognized'.format(error_type))
    errors = [error[error_type]]
    while not converged:
        a, b = compress(X, dim, saveA=False)
        res = solve(b, a, verbose=False, weight=5e-03, miniter=100, maxiter=2000)
        error = calc_error(X, res['lambdas'])[error_type]
        errors.append(error)
        grad = error - old_error
        print dim, error, grad
        if error < reconstruction_tolerance:
            converged = True
        elif grad < error_convergence:
            converged = True
        old_dim = deepcopy(dim)
        old_error = deepcopy(error)
        dim = dim+5
        if dim > feature_size_max:
            dim = old_dim
            converged = True
    return a, b, dim, errors

if __name__ == '__main__':
    import hickle as hkl
    import pylab as pl
    import seaborn as sns
    sns.set_context('poster')
    sns.set_style('darkgrid')
    sns.set_palette('Set2')
    data = hkl.load('/home/epyzerknapp/Projects/columbus/playground/cep_timing_test/cep_test_50k.hkl')
    inputs = data['512_morgans_r2'][:100].T
    a, b, dim, errors = optimize_compression(inputs, error_type='tanimoto')
    samples = range(0, len(errors))
    samples = [x*5 for x in samples]
    pl.plot(samples, errors)

    ax = pl.gca()
    ax.set_xticklabels(np.arange(0,204,5))
    pl.ylabel('Tanimoto Reconstruction Error')
    pl.xlabel('Number of features')
    pl.show()








