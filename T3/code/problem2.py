# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
from Perceptron import Perceptron
from numpy.core.umath_tests import inner1d


# Implement this class
class KernelPerceptron(Perceptron):
    def __init__(self, numsamples):
        self.numsamples = numsamples
        self.alphas =  {} #of len(x)
        self.b = 0.0
        self.SV_indices = []
        self.X = []
        self.Y = []

    # Implement this!
    def fit(self, X, Y):
        X = np.array(X)
        self.X = X
        self.Y = Y
        # K(x,x2 ) = x.T *  x2
        np.random.seed(314159)
        SV_indices = set()
        alphas = {}
        print('X shape', X.shape)
        
        for i in range(self.numsamples):
            t = np.random.randint(X.shape[0])
            xt = X[t]
            y_hat = np.sum(alphas[idx]*inner1d(xt,X[idx]) for idx in SV_indices)
            y_true = Y[t]
            if y_true * y_hat <= 0:
                SV_indices.add(t)
                alphas.update({t:y_true})
        self.alphas = alphas
        self.SV_indices = SV_indices
        print('indices', SV_indices)
        print('alphas', alphas)

    def predict(self, X):
        print("========= *** Predicting *** ==========")
        print('x shape', X.shape)
        print('x shape', self.X.shape)
        print('support vector indices', self.SV_indices)
        print('alphas', self.alphas)

        y_hats = []
        for xt in X:
            boo =  [self.alphas[sv_i]*np.dot(xt,self.X[sv_i]) for sv_i in self.SV_indices] 
            print('boo', boo)
            y_hat = np.sum(boo)
            y_hats.append(y_hat)
        y_hats = np.array(y_hats)
        y_hats = np.sign(y_hats)

        print(y_hats.shape)
        print('y_hats', y_hats)
        print('alphas', self.alphas)

        return y_hats 


# Implement this class
class BudgetKernelPerceptron(Perceptron):
    def __init__(self, beta, N, numsamples):
        self.beta = beta
        self.N = N
        self.numsamples = numsamples

    # Implement this!
    # TODO: after kernel perceptron is fixed, merge into budget perceptron
    def fit(self, X, Y):
        # update step
        if y_true * y_hat <= self.beta:
            SV_indices.append(t)
            alphas.update({t:y_true})
            # removal step
            if len(SV_indices) > self.N:
                foo = [ Y[i] * (y_hat[i] - alphas[i]*np.dot(X[i].T, x[i])) \
                        for i in SV_indices ]
                arg_maxmarg = foo.index(max(foo))
                SV_indices.pop(arg_maxmarg)


    # Implement this!
    # def predict(self, X):



# Do not change these three lines.
#data = np.loadtxt("data.csv", delimiter=',')
data = np.loadtxt("data_short.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0 #budget
N = 100 #budget
#numsamples = 20000
numsamples = 200

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
k = KernelPerceptron(numsamples)
k.fit(X,Y)
k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, \
        include_points=True)

bk = BudgetKernelPerceptron(beta, N, numsamples)
bk.fit(X, Y)
bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=True)
