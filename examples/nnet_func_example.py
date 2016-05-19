from maml import neural_networks as nn
import hickle
import numpy as np
import argparse
import pickle



dataset = 'func_nn_example_data.hkl'
ins = '1024_morgans_r2'

parser = argparse.ArgumentParser()
# TODO: Implement these functions
# parser.add_argument("-n", "--n_nets", type=int,
#     default=None, help="Specifies the index of a net you are training")
# parser.add_argument("-t", "--test_frac", type=float,
#     default=0.1, help="Specifies the test fraction")
parser.add_argument("-o", "--output", type=str,
    default='homos,lumos,pces', help="Specifies the outputs")
parser.add_argument('-p', '--plot', dest='plot',  help='Plot the results', default=False, action='store_true')
parser.add_argument('-c', '--ncores', dest='cores', help='Number of cores to use', default=4, type=int)
parser.add_argument('-nt', '--normalize-inputs', dest='normX', help='Normalize inputs', default=False, type=bool, action='store_true')
args = parser.parse_args()
outs = args.output.split(',')

data = hickle.load('data/' + dataset)
X = data[ins]
Y = np.array([data[attr] for attr in outs]).T



if args.normX:
    std_X = np.std(X, 0)
    std_X[ std_X == 0 ] = 1
    mean_X = np.mean(X, 0)
else:
    std_X = np.ones(X.shape[ 1 ])
    mean_X = np.zeros(X.shape[ 1 ])

X = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
                    np.full(X_train.shape, self.std_X_train)




Y_means = np.mean(Y,axis=0)
Y_stds = np.std(Y,axis=0)
Y = (Y - Y_means) / Y_stds

P = np.arange(X.shape[0])
np.random.shuffle(P)
n_split = X.shape[0]/20
P_train = P[n_split:]
P_test = P[:n_split]

X_train = X[P_train]
Y_train = Y[P_train]
X_test = X[P_test]
Y_test = Y[P_test]

nnet = nn.train(X_train,X_test,Y_train,Y_test,[128,64,32],'tanh',learn_rate=0.002,n_epochs=20000,n_cores=args.cores)
print np.mean(np.abs(Y_stds * (nn.output(X_test,nnet) - Y_test)),axis=0)
nnet['Y_means'] = Y_means
nnet['Y_stds'] = Y_stds


with open('func_example.pkl','w') as fp:
    pickle.dump(nnet,fp)
if args.plot:
    import pylab as p
    for i in range(3):
        pred = nn.output(X_test,nnet)[:,i]
        true = Y_test[:,i]
        # true = (true * net._target_ms[1][i]) + net._target_ms[0][i]
        p.scatter(true, pred)
        amin = min(np.min(pred),np.min(true))
        amax = max(np.max(pred),np.max(true))
        p.ylabel('Predicted')
        p.xlabel('Actual')
        p.xlim(amin, amax)
        p.ylim(amin, amax)
        p.show()



