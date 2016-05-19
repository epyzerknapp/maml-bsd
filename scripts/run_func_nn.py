from maml import neural_networks as nn
import hickle
import numpy as np
import pickle
from ConfigParser import SafeConfigParser
import sys
from sklearn import cross_validation as cv
input_file = sys.argv[1]



parser = SafeConfigParser()


parser.read(input_file)
# initial read in of values

io = dict(parser.items('io'))
pred = dict(parser.items('prediction'))
multicore = dict(parser.items('multicore'))

#lets sanitize our reads which should not be strings
pred['arch'] = [int(i) for i in pred['arch'].split(',')]
pred['epochs'] = int(pred['epochs'])
pred['learn_rate'] = float(pred['learn_rate'])
pred['training_split'] = int(pred['training_split'])
pred['batch_size'] = int(pred['batch_size'])
multicore['ncores'] = int(multicore['ncores'])

#lets read in the data
data = hickle.load(io['datafile'])
outs = pred['outputs'].split(',')
ins = pred['inputs']
X = data[ins]
Y = np.array([data[attr] for attr in outs]).T

print io

if pred['normalise_inputs'] == True:
    std_X = np.std(X, 0)
    std_X[ std_X == 0 ] = 1
    mean_X = np.mean(X, 0)
else:
    std_X = np.ones(X.shape[ 1 ])
    mean_X = np.zeros(X.shape[ 1 ])

X = (X - np.full(X.shape, mean_X)) / \
                    np.full(X.shape, std_X)




Y_means = np.mean(Y,axis=0)
Y_stds = np.std(Y,axis=0)
Y = (Y - Y_means) / Y_stds
try:
    seed = int(io['seed'])
except KeyError:
    seed = 99
split = 1 - (0.01 * pred['training_split'])
X_train, X_test, Y_train, Y_test = cv.train_test_split(X, Y, test_size=split, random_state=seed)

nnet = nn.train(X_train,X_test,Y_train,Y_test,pred['arch'],pred['activation'],learn_rate=pred['learn_rate'],n_epochs=pred['epochs'],n_cores=multicore['ncores'], batch_size=pred['batch_size'], threshold=0.01)
print np.mean(np.abs(Y_stds * (nn.output(X_test,nnet) - Y_test)),axis=0)
nnet['Y_means'] = Y_means
nnet['Y_stds'] = Y_stds


with open(io['resultsfile'],'w') as fp:
    pickle.dump(nnet,fp)

out = dict()
out['X_train'] = X_train
out['Y_train'] = Y_train
hklname = io['resultsfile'].split('.')[0]
hklname = 'results_{}.hkl'.format(hklname)
hkl.dump(out,hklname)
if io['plot'] == 'True':
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



