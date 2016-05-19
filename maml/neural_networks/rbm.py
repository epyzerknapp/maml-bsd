__author__ = 'epyzerknapp'
"""
GLSP, or greedy layerwise supervised pre-training, is a method for representation learning which utilizes neural
networks.  From the original features, a funnel of neurons reduces the dimensionality of the input with the loss
function for each layer based upon reproduction error from the previous layer.
"""

import numpy as np

def sigmoid(x):
    """
    The sigmoid activation fuction, implemented as a logistic function.
    :param x: inputs
    :return: sigmoid acting on x
    """
    activation = 1./(1. + np.exp(-1. * x))
    return activation

def grad(x):
    """
    The gradient (derivative) of the sigmoid function
    :param x: inputs
    :return: derivative of sigmoid, given x.
    """
    deriv = x * (1 - x)
    return deriv

def train_rbm(data, n_hidden, epsilon=1e-04, epochs=100, weightcost=2e-04, momentum_init=0.5, momentum_final=0.9):
    """
    Trains a restricted Boltzmann machine using the CD1 algorithm proposed by Hinton et at in which only one timestep i
    n the Markov chain is used, rather than training the chain to equilibirum. In future implementations, the length of
    the Markov chain will be allowed to vary, and in a dynamic manner.
    The momentum is updated in a two step process with the first 5 epochs being determined using the initial momentum,
    with a second momentum being applied after this initial phase.
    :param data: ndarray, the data which is used for training
    :param n_hidden: the number of hidden units
    :param epsilon: the epsilon parameter of gradient descent minimzer
    :param epochs: number of epochs of gradient descent
    :param momentum_init: Initial value for momentum
    :param momentum_final: Final value for momentum
    :return: Weights, bias in visible weights, bias in hidden weights
    """
    n_visible = data.shape[1]
    W = np.random.normal(size=(n_visible, n_hidden))
    biasH = np.random.normal(size=(1, n_hidden))
    biasX = np.random.normal(size=(1, n_visible))
    incW = 0
    incbiasH = 0
    incbiasX = 0
    for j in range(epochs):
        np.random.shuffle(data)
        error = 0
        if j > 5:
            momentum = momentum_final
        else:
            momentum = momentum_init
        i = 0
        for idata in data:
            idata = np.atleast_2d(idata)
            h_given_x_presigmoid = np.dot(idata, W) + biasH
            h_given_x = sigmoid(h_given_x_presigmoid)
            h1_sample = np.random.binomial(n=1, p=h_given_x, size=h_given_x.shape)
            x_given_h_presigmoid = np.dot(h1_sample, W.T) + biasX
            x_given_h = sigmoid(x_given_h_presigmoid)
            v1_sample = np.random.binomial(n=1, p=x_given_h, size=x_given_h.shape)
            h_given_x_2_presigmoid = np.dot(v1_sample, W) + biasH
            h_given_x_2 = sigmoid(h_given_x_2_presigmoid)
            incW = momentum * incW + (1 - momentum) * epsilon * (np.dot(idata.T, h_given_x) -
                                                                 np.dot(x_given_h.T, h_given_x_2)
                                                                 - weightcost * W)
            incbiasH = momentum * incbiasH + (1 - momentum) * epsilon * (h_given_x - h_given_x_2)
            incbiasX = momentum * incbiasX + (1 - momentum) * epsilon * (idata - x_given_h)
            i+=1
            W = W + incW
            biasH += incbiasH
            biasX += incbiasX
        error = 0
        for idata in data:
            idata = np.atleast_2d(idata)
            phx = sigmoid(biasH + np.dot(idata, W))
            pxh = sigmoid(biasX + np.dot(phx, W.T))
            error += np.linalg.norm(idata - pxh, 2) / np.linalg.norm(idata, ord=2)
        print error / data.shape[0]
    return W, biasH, biasX




if __name__ == '__main__':
    import hickle as hkl
    data = hkl.load('/home/epyzerknapp/Projects/columbus/playground/cep_timing_test/cep_test_50k.hkl')
    inputs = data['512_morgans_r2'][:1000]
    train_rbm(inputs, 100)




