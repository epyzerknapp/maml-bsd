__author__ = 'epyzerknapp'
import numpy as np
from numpy.linalg import cholesky
#from scipy.linalg import cholesky
from scipy.linalg import cho_solve
from scipy.spatial.distance import pdist, squareform
import os,sys
import cPickle
from scipy.optimize import minimize


class GaussianProcess:
    """
    The holder class for holding the gaussian process technique
    """

    def __init__(self, features_train, target_train, features_test, targets_test=None, kernel_used='tanimoto'):

        if type(features_train) == np.ndarray:
            self.features_train = features_train
        else:
            raise TypeError("Training features must be of type numpy ndarray")
        if type(features_test) == np.ndarray:
            self.features_test = features_test
        else:
            raise TypeError("Testing features must be of type numpy ndarray")
        if type(target_train) == np.ndarray:
            self.targets_train = target_train
        else:
            raise TypeError("Training targets must be of type numpy ndarray")
        if type(targets_test) == np.ndarray or targets_test is None:
            self.targets_test = targets_test
        else:
            raise TypeError("Features must be of type numpy ndarray")
        if targets_test is None:
            self.allow_goodness = False
        else:
            self.allow_goodness = True
        self.custom_kernel = None
        self.available_kernels = {'tanimoto':self._default_kernel, 'gaussian':self._gaussian_kernel,
            'laplacian':self._laplace_kernel,'hamming':self._hamming_distance,'custom':self._custom_kernel} #make sure to keep default kernel as first entry!
        self.amp = 1
        self.std = 1
        self.noise = None
        self.goodness = {}
        self.predicted_targets = None
        self.predicted_uncertainty = None
        self.cholesky = None
        self.train_cholesky = None
        if kernel_used not in self.available_kernels.keys():
            raise NotImplementedError("{} is not a supported kernel.  Please use one of the following:\n{}".format(kernel_used,
            "\n".join(self.available_kernels.keys())))
        self.kernel_used = kernel_used
        self.optimization_target = "rms_error"
        self.invert_answer=False
        self.minimizer = "BFGS"
        self.bounds = None

    def kernel(self, x1, x2):
        """
        This is a wrapper for the kernels
        """
        return self.available_kernels[self.kernel_used](x1,x2)

    def _default_kernel(self, x1, x2):
        """
        This is a function that takes in two sets of features, and computes the co-variance matrix

        **Variables**
        x1, x2 : np.ndarray : feature sets with shape (N1, D) and (N2, D) respectively

        **Returns**
        covar : np.ndarray : covariance matrix with dimensions (N1, N2)

        """

        amp = self.amp
        noise = self.noise
        return amp * self.tanimoto_similarity(x1, x2) + self._noise_kernel(x1, x2, noise)

    def _noise_kernel(self, x1, x2, noise):
        """
        This tests whether x1 and x2 refer to the same python object,
        not just that they are equal (also much faster)
        """
        if x1 is x2:
        # if x1.shape == x2.shape and (x1 == x2).all():
            return noise**2*np.eye(x1.shape[0])
        else:
            return np.zeros((x1.shape[0],x2.shape[0]))

    def set_custom_kernel(self, kernel):
        """
        This allows the setting of a custom kernel
        """
        self.kernel_used = 'custom'
        self.custom_kernel = kernel

    def _custom_kernel(self, x1, x2):
        """
        This is a wrapper for the kernels
        """
        if callable(self.custom_kernel):
            return self.custom_kernel(self, x1, x2)
        else:
            raise Exception('No custom kernel set')

    def _gaussian_kernel(self, x1, x2):
        # calculate squared distances, using scipy pdist speedup if x1 == x2
        sq_norm = (squareform(pdist(x1, 'sqeuclidean')) if x1 is x2 else
                np.sum(np.power(x1[:,None,:] - x2[None,:,:],2),axis=2))

        gk = pow(self.amp,2) * np.exp(sq_norm / (-2 * pow(self.std,2)))
        return gk + self._noise_kernel(x1, x2, self.noise)

    def _laplace_kernel(self, x1, x2):
        # calculate squared distances, using scipy pdist speedup if x1 == x2
        norm = np.linalg.norm(x1[:,None,:] - x2[None,:,:],axis=2)

        gk = pow(self.amp,2) * np.exp(norm / (-2 * pow(self.std,2)))
        return gk + self._noise_kernel(x1, x2, self.noise)

    def _hamming_distance(self, x1, x2):
        norm = np.linalg.norm(x1[:,None,:] - x2[None,:,:],axis=2)
        return self.amp*norm + self._noise_kernel(x1, x2, self.noise)

    def set_minimizer(self, minimizer, bounds=None):
        """
        This allows the setting of one of the minimizers from scipy for use in hyperparameter optimization
        """
        allowed_minimizers = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC',
        'COBYLA', 'SLSQP', 'trust-ncg']

        if minimizer not in allowed_minimizers:
            raise NotImplementedError("{} is not a supported minimizer.  Please use one of the following:\n{}".format(minimizer,
            "\n".join(allowed_minimizers)))
        self.minimizer = minimizer
        self.bounds = bounds

    def _guess_amp_and_noise(self,reduced_set):
        """
        This is an internal function that creates a good 'first guess' for the noise and amplitude hyperparameters.
        It uses the relationship:
        amplitude_guess = std(targets)
        noise_guess = 0.1 * amplitude_guess

        **Variables**
        targets: np.ndarray : set of values to train against


        **Returns**
        amp, noise : tuple of the first guess for amplitude and noise
        """
        if self.kernel_used == 'tanimoto' or self.kernel_used == 'hamming':
            amp_guess = np.std(self.targets_train)
            noise_guess = 0.1 * amp_guess
            std_guess = 1.0
        elif self.kernel_used == 'gaussian' or self.kernel_used == 'laplacian':
            #amp_guess = 1.0
            amp_guess = np.std(self.targets_train)
            noise_guess = 0.1 * amp_guess
            x = self.features_train if not self.reduced_set else self.reduced_set[0]
            std_guess = np.std(x)
        else:
            amp_guess = 1.0
            noise_guess = 0.01
            std_guess = 1.0

        self.amp = amp_guess
        self.noise = noise_guess
        self.std = std_guess

    def optimize_noise_and_amp(self, reduced_set=None, optimize_on="rms_error", invert_answer=False):
        """
        This function optimizes the values for noise and amplitude based upon rms error
        """
        self.reduced_set = reduced_set
        self._guess_amp_and_noise(reduced_set)
        self.optimization_target = optimize_on
        self.invert_answer=invert_answer
        print "Optimizing Hyperparameters over noises, and amps using {}".format(self.minimizer)
        if optimize_on == "log_marg_lik" and self.kernel_used == "gaussian" and self.targets_train.shape[1] == 1:
            res = minimize(self.optimize_gp, [self.noise, self.amp, self.std], options={'disp': True}, method=self.minimizer,
                bounds=self.bounds, jac=self._gaussian_log_marg_lik_grad)
        elif optimize_on == "log_marg_lik" and self.kernel_used == "laplacian" and self.targets_train.shape[1] == 1:
            res = minimize(self.optimize_gp, [self.noise, self.amp, self.std], options={'disp': True}, method=self.minimizer,
                bounds=self.bounds, jac=self._laplacian_log_marg_lik_grad)
        else:
            res = minimize(self.optimize_gp, [self.noise, self.amp, self.std], options={'disp': True}, method=self.minimizer,
                bounds=self.bounds)
        print res
        print "Optimized parameters are -\nnoise:{}\namplitude:{}".format(self.noise, self.amp)

    def optimize_gp(self, params):
        """
        A wrapper for training the hyperparameters
        """
        self.noise = params[0]
        self.amp = params[1]
        self.std = params[2]
        rms = self.run_gp(reduced_set=self.reduced_set, training_mode=True, force_cholesky=True)
        return rms

    def _gaussian_log_marg_lik_grad(self, params):
        self.noise = params[0]
        self.amp = params[1]
        self.std = params[2]
        x = self.features_train if not self.reduced_set else self.reduced_set[0]
        y = self.targets_train if not self.reduced_set else self.reduced_set[2]
        Kxx = self.kernel(x, x)
        Kxx_p = [2* self.noise * np.eye(x.shape[0])]
        sq_norm = np.sum(np.power(x[:,None,:] - x[None,:,:],2), axis=2)
        Kxx_p.append(2* self.amp * np.exp(- sq_norm/(2 * pow(self.std,2))))
        Kxx_p.append((sq_norm/pow(self.std,3)) * \
            pow(self.amp,2) * np.exp(- sq_norm/(2*pow(self.std,2))))
        KxxI = np.linalg.inv(Kxx)
        grad = []
        for Kxx_pi in Kxx_p:
            # print KxxI.shape, Kxx_pi.shape, y.shape
            grad.append(0.5* (-np.trace(np.dot(KxxI,Kxx_pi)) + np.dot(np.dot(y[:,0],np.dot(np.dot(KxxI,Kxx_pi),KxxI)),y[:,0] )))
        return -np.array(grad)

    def _laplacian_log_marg_lik_grad(self, params):
        self.noise = params[0]
        self.amp = params[1]
        self.std = params[2]
        x = self.features_train if not self.reduced_set else self.reduced_set[0]
        y = self.targets_train if not self.reduced_set else self.reduced_set[2]
        Kxx = self.kernel(x, x)
        Kxx_p = [2* self.noise * np.eye(x.shape[0])]
        norm = np.linalg.norm(x[:,None,:] - x[None,:,:], axis=2)
        Kxx_p.append(np.exp(- norm/(2 * pow(self.std,2))))
        Kxx_p.append((norm/pow(self.std,3)) * \
            self.amp * np.exp(- norm/(2*pow(self.std,2))))
        KxxI = np.linalg.inv(Kxx)
        grad = []
        for Kxx_pi in Kxx_p:
            # print KxxI.shape, Kxx_pi.shape, y.shape
            grad.append(0.5* (-np.trace(np.dot(KxxI,Kxx_pi)) + np.dot(np.dot(y[:,0],np.dot(np.dot(KxxI,Kxx_pi),KxxI)),y[:,0] )))
        return -np.array(grad)

    def _gaussian_log_prob(self, L, y):
        """
        Log prob of values y according to a Gaussian distribution
        L is the lower cholesky decomposition of the covariance matrix
        y has to be of size (N,1)
        """
        a = cho_solve((L, True), y)
        nll = - np.sum(np.log(np.diag(L))) - 0.5 * (y.T.dot(a))[0,0] -0.5*self.features_train.shape[0]*np.log(2*np.pi)
        # W = cho_solve((L, True), np.eye(len(self.features_train))) - a*a.T
        # ders = np.zeros(2)
        # for i in [0,1]:
        #     ders[i] = np.sum(W*)
        return nll

    def compute_cholesky(self, x, train=False):
        """
        Computes the cholesky decomposition

        """
        Kxx = self.kernel(x, x)
        # import pylab as pl
        # pl.imshow(Kxx)
        # pl.show()
        if not train:
            self.cholesky = cholesky(Kxx)
        else:
            self.train_cholesky = cholesky(Kxx)

    def _recall(self, y1, y2, cutoff_percent):
        """
        computes the fraction of the top `cutoff_percent` of y1 lie in
        the top `cutoff percent` of y1
        """
        top_1 = y1 > np.percentile(y1, 100 - cutoff_percent)
        top_2 = y2 > np.percentile(y2, 100 - cutoff_percent)
        return float(np.sum(top_1*top_2))/np.sum(top_1)

    def tanimoto_similarity(self, x1, x2):
        """
        Computes tanimoto distance between each pair of points in x1 and x2
        x1 : (N1, D) array of binary features (0 or 1)
        x2 : similar
        returns: (N1, N2) matrix of pairwise similarities (ratio of intersection to union)
        """
        def isbinary(x):
            return np.all(np.logical_or(x == 1, x == 0))
        if not isbinary(x1) and isbinary(x2) and x1.shape[1] == x2.shape[2]:
            raise ValueError("X1 and X2 must be binaries, with the same shape")
        x1 = x1.astype('float')
        x2 = x2.astype('float')
        S1 = np.sum(x1, axis=1)  # size of each x1
        S2 = np.sum(x2, axis=1)  # size of each x2
        I = np.dot(x1, x2.T)  # intersection of x1 and x2

        denominator = (S1[:, None] + S2[None, :] - I)
        zeros = denominator == 0    # <--- This shouldn't be necessary for the not-zero fingerprint.
        denominator[zeros] = 1  # avoid divide-by-zero warning
        similarity_matrix = I.astype(float)/denominator
        similarity_matrix[zeros] = 1  # enforce tanimoto_similarity([0,0,0],[0,0,0]) = 1
        return similarity_matrix

    def _sanitize(self,X):
        def is_pos_def(x):
            # print np.linalg.eigvals(x)
            try:
                ans = np.all(np.linalg.eigvals(x) > 0)
            except np.linalg.LinAlgError:
                return False
            return ans
        def _rec_sanitize(X,tresh):
            sane = np.where(X>tresh,X,np.zeros(X.shape))
            if np.all(sane == 0): return sane
            if is_pos_def(sane):
                return sane
            else:
                return _rec_sanitize(X,tresh*1.1)
        return _rec_sanitize(X,0.01*np.mean(X))

    def run_gp(self, reduced_set=None, training_mode=False, force_cholesky=False):
        """
        Computes posterior GP mean and covariance at points xt based on the training data x, t
        x : numpy nparray, required :(N, D) array of features stored as self.features_train
        y : numpy nparray, required :(N, K) vector of targets stored as self.targets_train
        xt: numpy nparray, required :(Nt, D) array of test set features stored as self.features_test
        yt: numpy nparray, optional :(Nt, K) stored as self.features_test

        reduced_set : list, optional : a smaller set of testing and training features for optimizing the noise
        and amplitude hyperparameters.
        training_mode : bool : flag to set the return of a measure of goodness
        force_cholesky: bool : flag to force the new computation of the cholesky decomposition
        kernel: function, required : function to compute Gram/covariance matrix, stored as self.kernel
        NOTE: Using explicit noise is deprecated. Please use implicit noise in your kernel
        (e.g. using the noise_kernel built into this class)
        diag_kernal : function, optional : function that computes diagonal of covariance matrix
        L : Cholesky decomposition matrix (optional) stored as self.cholesky
        Avaliable measures of 'goodness':
        * test set rms error
        * test set mean absolute error
        * log marginal likelihood
        * predictive log prop

        """


        if not reduced_set:
            x = self.features_train
            xt = self.features_test
            y = self.targets_train
            yt = self.targets_test
        else:
            x = reduced_set[0]
            xt = reduced_set[1]
            y = reduced_set[2]
            yt = reduced_set[3]


        Kxt = self.kernel(x, xt)
        Ktt = self.kernel(xt, xt)

        #N, D = x.shape
        #assert(xt.shape[1] == D)
        y_mean = np.mean(y)
        if not force_cholesky:
            if not training_mode:
                if self.cholesky is None:
                    self.compute_cholesky(x)
                    L = self.cholesky
                else:
                    L = self.cholesky
            else:
                if self.train_cholesky is None:
                    self.compute_cholesky(x, train=True)
                    L = self.train_cholesky
                else:
                    L = self.train_cholesky
        else:
            if not training_mode:
                self.compute_cholesky(x)
                L = self.cholesky
            else:
                self.compute_cholesky(x, train=True)
                L = self.train_cholesky

        solve = cho_solve((L, True), Kxt)
        gp_mean = solve.T.dot(y-y_mean)
        gp_covar = Ktt - Kxt.T.dot(solve)
        y_pred = gp_mean+y_mean
        y_std = np.sqrt(np.diag(gp_covar))
        Lt = cholesky(gp_covar)

        if self.allow_goodness:
            goodness = {}
            goodness['rms_error'] = np.std(y_pred - yt)
            goodness['frac_recall'] = self._recall(yt, y_pred, cutoff_percent=5)
            goodness['log_marg_lik']  = self._gaussian_log_prob(L, y - y_mean)
            goodness['log_pred_prob'] = self._gaussian_log_prob(Lt, yt - y_pred)
            self.goodness = goodness
        self.predicted_targets = y_pred
        self.predicted_uncertainty = y_std
        print "noise: {} amp: {} std: {} {}".format(self.noise, self.amp, self.std, " ".join(["{}: {}".format(item[0], item[1]) for item in self.goodness.items()]))
        if training_mode and self.allow_goodness:
          if self.invert_answer:
            return -1. * self.goodness[self.optimization_target]
          else:
            return self.goodness[self.optimization_target]

    def load_new_testing_data(self, features_test, targets_test):
        """
        Allows overwriting of testing (predicted) feature sets. This enables new results without recomputing the
        cholesky decomp for the training data.
        """
        self.features_test = features_test
        self.targets_test = targets_test

    def save_as_pickle(self, filename):
        """
        This method allows you to dump the GaussianProcess object as a pickel, allowing you to reload later, without
        needing to repeat the training and hyper-parameter optimization.
        """
        with open(filename, "w") as f:
            cPickle.dump(self, f, 2)

    @classmethod
    def loader(cls, filename):
        """
        This loader allows you to reload a dumped class.
        You use it as follows:
        gp = gp.loader(filename)
        """
        with open(filename) as f:
            return cPickle.load(f)
