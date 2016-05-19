__author__ = 'epyzerknapp', 'tmarkovich'

import numpy as np
import time
import numba
from numba import f8, b1, u1

# TODO: Cleanup code for release.
class TwistSolver(object):

    def __init__(self, psi_function=None, phi_function=None, lam1=1e-04, stop_criterion=1, tolA=1e-04, debias=1,
                 tolD=1e-05, maxiter=1000, miniter=5, maxiter_debias=200,
                 miniter_debias=5, init=0, weight=1e-03,verbose=True):
        """
        Initializes the settings for the TwistSolver class.

        TwIST (Two-step Iterative Shrinkage/Thresholding Algorithm for Linear Inverse Problems)

        The TwIST minimizer called here is described at:
        http://www.lx.it.pt/~bioucas/TwIST/TwIST.htm
        and in the paper:
        A New TwIST: Two-Step Iterative Shrinkage Thresholding Algorithms for Image Restoration
        by Jose M. Bioucas-Dias and Mario A. T. Figueiredo

        In TwIST, the update equation depends on the two previous estimates (thus, the term two-step), rather than
        only on the previous one. This class of minimizers contains and extends the iterative shrinkage/thresholding
        methods.

        :param psi_function: function handle, optional handle to denoising function (the default is soft threshold)
        :param phi_function:function handle, optional handle to regularizer needed to compute the objective function.
            (the default = :math:`||x||_1`)
        :param lam1:float, optional (default=0.04) parameter of the  TwIST algorithm: Optimal choice:
                ``lam1`` = min eigenvalue of ::math:`A^T*A`.
                If min eigenvalue of :math:`A^T*A` equals 0, or unknwon, set ``lam1`` to a value much smaller than 1.
                Rule of Thumb:
                * ``lam1=1e-4`` for severyly ill-conditioned problems
                * ``lam1=1e-2`` for mildly  ill-conditioned problems
                * ``lam1=1``    for A unitary direct operators
                .. note:: If max eigenvalue of ::math:`A^T*A` > 1,
               the algorithm may diverge. This is to be avoided
               by taking one of the follwoing  measures:
               1. Set ``enforce_monotone=True`` (default)
               2. Solve the equivalenve minimization problem
               .. math::
                   min_x = 0.5*|| (y/c) - (A/c) x ||_2^2 + (tau/c^2) \phi( x ),
               where :math:`c > 0` ensures that  max eigenvalue of ::math:`(A^TA/c^2) \leq 1`.
        :param stop_criterion: {0, 1, 2, 3}, optional (default=0) type of stopping criterion to use
                * ``stop_criterion=0`` algorithm stops when the relative change in the numbsolver = twist.TwistSolver(tolA=tolA, tolD=tolD, verbose=verbose, **kwargs)er
                                    of non-zero components of the estimate falls below ``tolA``
                * ``stop_criterion=1`` stop when the relative change in the objective function falls below ``tolA``
                * ``stop_criterion=2`` stop when the relative norm of the
                                    difference between two consecutive estimates falls below ``tolA``
                * ``stop_criterion=3`` stop when the objective function becomes equal or less than ``tolA``.
        :param tolA: float, optional (default=0.01) stopping threshold.
        :param debias: bool, optional (default=False) debiasing option
                note:: Debiasing is an operation aimed at the computing the solution of the LS problem
                math::
                arg min_x = 0.5*|| y - A^T x ||_2^2
                where ::math:`A^T` is the  submatrix of ``A`` obatained by deleting the columns of
                ``A ``corresponding of components of ``x`` set to zero by the TwIST algorithm

        :param tolD: float, optional (default=0.0001) stopping threshold for the debiasing phase.
                If no debiasing takes place, this parameter, is ignored.
        :param maxiter:int, optional (default=1000) maximum number of iterations allowed in
                the main phase of the algorithm.
        :param miniter:int, optional (default=5) minimum number of iterations performed in
                the main phase of the algorithm.
        :param maxiter_debias:int, optional (default=5) maximum number of iterations allowed in the
                debising phase of the algorithm.
        :param miniter_debias:int, optional (default=5) minimum number of iterations to perform in the
                debiasing phase of the algorithm.
        :param init: {0, 1, 2, array}, optional (default=0) must be one of
                * ``init=0`` Initialization at zero.
                * ``init=1`` Random initialization.
                * ``init=2`` initialization with ::math:`A^Ty`.
                * ``init=array`` initialization provided by the user
        :param weight: float, optional
        :param verbose: bool, optional (default=False) work silently (False) or verbosely (True)

        """
        self.psi_function = psi_function
        self.phi_function = phi_function
        self.lam1 = lam1
        self.stop_criterion = stop_criterion
        self.tolA = tolA
        self.debias = debias
        self.tolD = tolD
        self.maxiter = maxiter
        self.miniter = miniter
        self.maxiter_debias = maxiter_debias
        self.miniter_debias = miniter_debias
        self.init = init
        self.weight = weight
        self.verbose = verbose

    def solve(self, signal, A):
        tau = self.weight * np.max(np.abs(np.dot(A.T, signal.T)))


        x, x_debias, objective, times, debias_start, max_svd = TwIST(signal.T,
                                                                     A,
                                                                     tau,
                                                                     self.psi_function,
                                                                     self.phi_function,
                                                                     self.lam1,
                                                                     self.stop_criterion,
                                                                     self.tolA,
                                                                     self.debias,
                                                                     self.tolD,
                                                                     self.maxiter,
                                                                     self.miniter,
                                                                     self.maxiter_debias,
                                                                     self.miniter_debias,
                                                                     self.init,
                                                                     self.verbose)
        return x, x_debias, objective, times, debias_start, max_svd


# @numba.jit
def TwIST(y, A, tau, psi_function, phi_function, lam1, stop_criterion, tolA, debias, tolD,
          maxiter, miniter, maxiter_debias, miniter_debias, init, verbose):
    """

    y : array,
           1D vector or 2D array (image) of observations.

        A : {array, function handle},
            if y and x are both 1D vectors, ``A`` can be a
            k*n (where k is the size of ``y`` and n the size of ``x``)
            matrix or a handle to a function that computes
            products of the form :math:`Av`, for some vector v.
            In any other case (if ``y`` and/or ``x`` are 2D arrays),
            ``A`` has to be passed as a handle to a function which computes
            products of the form :math:`Ax`; another handle to a function
            ``AT`` which computes products of the form :math:`A^Tx` is also required
            in this case. The size of x is determined as the size
            of the result of applying ``AT``.

        tau : float,
            regularization parameter, usually a non-negative real
            parameter of the objective function (see above).

    """
    max_svd = 1
    debias_start = 0
    x_debias = []
    # twist parameters
    lamN = 1
    rho0 = (1 - lam1 / lamN) / (1 + lam1 / lamN)
    alpha = 2 / (1 + np.sqrt(1 - rho0 ** 2.))
    beta = alpha * 2 / (lam1 + lamN)
    AT = A.T
    Aty = np.dot(AT, y)
    if psi_function is None:
        psi_function = softThreshold
        psi_soft = True
    else:
        psi_soft = False
    if phi_function is None:
        phi_function = lambda x: np.sum(np.abs(x))
        phi_l1 = True
    else:
        phi_l1 = False
    max_tau = np.max(np.abs(Aty))
    if init == 0:
        x = np.dot(AT, np.zeros(y.shape))
    elif init == 1:
        # initialize randomly, using AT to find the size of x
        x = np.random.randn(AT(np.zeros(y.shape)).shape)
    elif init == 2:
        # initialize x0 = A'*y
        x = Aty
    else:
        raise Exception("Unknown 'Initialization' option")
    if not np.isscalar(tau) and tau.shape != x.shape:
        raise Exception('Parameter tau has wrong dimensions; it should be scalar or size(x)')
    nz_x = x != 0
    num_nz_x = float(np.sum(nz_x))

    resid = y - np.dot(A, x)
    prev_f = 0.5 * np.linalg.norm(resid.ravel()) ** 2 + tau * phi_function(x)

    # start the clock
    t0 = time.time()
    times = [0]
    objective = [prev_f]

    iter = 1
    if verbose:
        print '\nInitial objective = %10.6e,  nonzeros=%7d' % (prev_f, num_nz_x)
    # variables controling first and second order iterations
    IST_iters = 0
    TwIST_iters = 0
    xm2 = x
    xm1 = x
    cont_outer = True
    while cont_outer:
        # gradient
        grad = np.dot(AT, resid)
        while True:
            # IST estimate
            x = psi_function(xm1 + grad / max_svd, tau / max_svd)
            if (IST_iters >= 2) or (TwIST_iters != 0):
                # set to zero the past when the present is zero
                # suitable for sparse inducing priors
                xm1[x == 0] = 0
                xm2[x == 0] = 0
                # two-step iteration
                xm2 = (alpha - beta) * xm1 + (1 - alpha) * xm2 + beta * x
                # compute residual
                resid = y - np.dot(A, xm2)
                f = 0.5 * np.linalg.norm(resid.ravel()) ** 2 + tau * phi_function(xm2)
                if (f > prev_f):
                    # do a IST iteration if monotonocity fails
                    TwIST_iters = 0
                else:
                    # TwIST iterations
                    TwIST_iters += 1
                    IST_iters = 0
                    x = xm2
                    if TwIST_iters % 10000 == 0:
                        max_svd *= 0.9
                    break
            else:
                resid = y - np.dot(A, x)
                f = 0.5 * np.linalg.norm(resid.ravel()) ** 2 + tau * phi_function(x)
                if f > prev_f:
                    # if monotonicity  fails here  is  because
                    # max eig (A'A) > 1. Thus, we increase our guess
                    # of max_svs
                    max_svd *= 2
                    if verbose:
                        print 'Incrementing S=%2.2e' % max_svd
                    IST_iters = 0
                    TwIST_iters = 0
                else:
                    TwIST_iters += 1
                    # break while loop
                    break
        xm2 = xm1
        xm1 = x
        # Update the number of nonzero components and its variation
        nz_x_prev = nz_x
        nz_x = x != 0
        num_nz_x = np.sum(nz_x)
        num_changes_active = np.sum(nz_x != nz_x_prev)
        # take no less than miniter and no more than maxiter iterations
        if stop_criterion == 0:
            # compute the stopping criterion based on the change
            # of the number of non-zero components of the estimate
            criterion = num_changes_active
        elif stop_criterion == 1:
            # compute the stopping criterion based on the relative
            # variation of the objective function.
            criterion = np.abs(f - prev_f) / prev_f
        elif stop_criterion == 2:
            # compute the stopping criterion based on the relative
            # variation of the estimate.
            criterion = np.linalg.norm((x - xm1).ravel()) / np.linalg.norm(x.ravel())
        elif stop_criterion == 3:
            # continue if not yet reached target value tolA
            criterion = f
        else:
            raise Exception('Unknwon stopping criterion');
        cont_outer = (iter <= maxiter) and (criterion > tolA)
        if iter <= miniter:
            cont_outer = True
        iter += 1
        prev_f = f
        objective.append(f)
        times.append(time.time() - t0)
        # Print out the various stopping criteria
        if verbose:
            print 'Iteration=%4d, objective=%9.5e, nz=%7d,  criterion=%7.3e' % (iter, f, num_nz_x, criterion / tolA)
    if verbose:
        print '\nFinished the main algorithm!\nResults:'
        print '||A x - y ||_2 = %10.3e' % np.sum(resid * resid)
        print '||x||_1 = %10.3e' % np.sum(np.abs(x))
        print 'Objective function = %10.3e' % f
        print 'Number of non-zero components = %d' % num_nz_x
        print 'CPU time so far = %10.3e' % times[-1]
    # --------------------------------------------------------------
    # NOTE: If the 'Debias' option is set to 1, we try to
    # remove the bias from the l1 penalty, by applying CG to the
    # least-squares problem obtained by omitting the l1 term
    # and fixing the zero coefficients at zero.
    # --------------------------------------------------------------
    if debias:
        if verbose:
            print '\nStarting the debiasing phase...\n'
        x_debias = x.copy()
        debias_start = iter
        # calculate initial residual
        resid = np.dot(A, x_debias)
        resid = resid - y
        rvec = np.dot(AT, resid)
        # mask out the zeros
        zeroind = x_debias == 0
        rvec[zeroind] = 0
        rTr_cg = np.sum(rvec * rvec)
        # Set convergence threshold for the residual || RW x_debias - y ||_2
        tol_debias = tolD * rTr_cg
        # initialize pvec
        pvec = -rvec
        # main loop
        cont_debias_cg = True
        while cont_debias_cg:
            # calculate A*p = Wt * Rt * R * W * pvec
            RWpvec = np.dot(A, pvec)
            Apvec = np.dot(AT, RWpvec)
            # mask out the zero terms
            Apvec[zeroind] = 0
            # calculate alpha for CG
            alpha_cg = rTr_cg / np.sum(pvec * Apvec)
            # take the step
            x_debias = x_debias + alpha_cg * pvec
            resid = resid + alpha_cg * RWpvec
            rvec = rvec + alpha_cg * Apvec
            rTr_cg_plus = np.sum(rvec * rvec)
            beta_cg = rTr_cg_plus / rTr_cg
            pvec = -rvec + beta_cg * pvec
            rTr_cg = rTr_cg_plus
            iter += 1
            objective.append(0.5 * np.sum(resid * resid) + tau * phi_function(x_debias))
            times.append(time.time() - t0)
            # in the debiasing CG phase, always use convergence criterion
            # based on the residual (this is standard for CG)
            if verbose:
                print ' Iter = %5d, debias resid = %13.8e, convergence = %8.3e' % \
                      (iter, np.sum(resid * resid), rTr_cg / tol_debias)
            cont_debias_cg = \
                (iter - debias_start <= miniter_debias) or \
                ((rTr_cg > tol_debias) and (iter - debias_start <= maxiter_debias))
        if verbose:
            print '\nFinished the debiasing phase!\nResults:'
            print '||A x - y ||_2 = %10.3e' % np.sum(resid * resid)
            print '||x||_1 = %10.3e' % np.sum(np.abs(x))
            print 'Objective function = %10.3e' % f

            nz = x_debias != 0
            print 'Number of non-zero components = %d' % np.sum(nz)
            print 'CPU time so far = %10.3e\n' % times[-1]

    return x, x_debias, objective, times, debias_start, max_svd


@numba.jit
def softThreshold(x, threshold):
    """
    Apply Soft Thresholding

    Parameters
    ----------

    x : array-like
        Vector to which the soft thresholding is applied

    threshold : float
        Threhold of the soft thresholding

    Returns:
    --------
    y : array
        Result of the applying soft thresholding to x.

        .. math::

            y = sign(x) \star \max(\abs(x)-threshold, 0)
    """
    y = np.abs(x) - threshold
    y[y < 0] = 0
    y[x < 0] = -y[x < 0]
    return y


@numba.jit
def hardThreshold(x, threshold):
    """
    Apply Hard Thresholding

    Parameters
    ----------

    x : array-like
        Vector to which the hard thresholding is applied

    threshold : float
        Threhold of the hard thresholding

    Returns:
    --------
    y : array
        Result of the applying hard thresholding to x.

        .. math::

            y = x * (\abs(x) > threshold)
    """
    y = np.zeros_like(x)
    ind = np.abs(x) > threshold
    y[ind] = x[ind]
    return y
