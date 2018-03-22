# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
import time
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

    def fit(self, X, Y):
        print("\n\n========= *** Fitting Perceptron *** ==========")
        t_start = time.time()
        X = np.array(X)
        self.X = X
        self.Y = Y
        # K(x,x2 ) = x.T *  x2
        #np.random.seed(314159)
        SV_indices = set()
        alphas = {}
        
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
        t_elapsed = time.time() - t_start
        print("Elapsed time for fitting ", X.shape, " datapoints: ", t_elapsed, " seconds." )

    def predict(self, X):
        t_start = time.time()
        print("========= *** Predicting *** ==========")
        print('support vector indices', self.SV_indices)
        print('\nalphas', self.alphas)

        y_hats = []
        for xt in X:
            boo =  [self.alphas[sv_i]*np.dot(xt,self.X[sv_i]) for sv_i in self.SV_indices] 
            #print('boo', boo)
            y_hat = np.sum(boo)
            if y_hat == 0: #dealing with the meshgrid has zeros at the beginning
                y_hat = 1
            y_hats.append(y_hat)
        y_hats = np.array(y_hats)
        y_hats = np.sign(y_hats)

        t_elapsed = time.time() - t_start
        print("Elapsed time for predicting", X.shape, " datapoints: ", t_elapsed, "seconds.")
        return y_hats 


# Implement this class
class BudgetKernelPerceptron(Perceptron):
    def __init__(self, beta, N, numsamples):
        self.beta = beta
        self.N = N
        self.numsamples = numsamples
        self.X = X
        self.Y = Y

    def fit(self, X, Y):
        print("\n\n========= *** Fitting Budget Perceptron *** ==========")
        t_start = time.time()

        X = np.array(X)
        self.X = X
        self.Y = Y

        SV_indices = []
        alphas = {}
        y_hats = {}

        for i in range(self.numsamples):
            t = np.random.randint(X.shape[0])
            xt = X[t]
            y_hat = np.sum(alphas[idx]*inner1d(xt,X[idx]) for idx in SV_indices)
            y_hats.update({t:y_hat})
            y_true = Y[t]

            if y_true * y_hat <= self.beta:
                if t not in SV_indices:
                    SV_indices.append(t)
                alphas.update({t:y_true})
                # removal step
                if len(SV_indices) > self.N:
                    foo = [ Y[i] * (y_hats[i] - alphas[i]*np.dot(X[i], X[i])) \
                            for i in SV_indices ]
                    arg_maxmarg = foo.index(max(foo))
                    #print('remove', arg_maxmarg)
                    #print('sV_indices', SV_indices)
                    del(SV_indices[arg_maxmarg])
        self.alphas = alphas
        self.SV_indices = SV_indices
        t_elapsed = time.time() - t_start
        print("Elapsed time for fitting ", X.shape, " datapoints: ", t_elapsed, " seconds." )

    def predict(self, X):
        t_start = time.time()
        print("========= *** Predicting *** ==========")
        #print('support vector indices', self.SV_indices)
        #print('alphas', self.alphas)

        y_hats = []
        for xt in X:
            y_hat = np.sum(boo)
            if y_hat == 0: #dealing with the meshgrid has zeros at the beginning
                y_hat = 1
                print('\nY_hat is zero. boo: ', boo, ' xt: ', xt)
            y_hats.append(y_hat)
        y_hats = np.array(y_hats)
        y_hats = np.sign(y_hats)

        t_elapsed = time.time() - t_start
        print("Elapsed time for predicting", X.shape, " datapoints: ", t_elapsed, "seconds.")
        return y_hats 


# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
#data = np.loadtxt("data_short.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0 #budget
N = 100 #budget
numsamples = 20000
#numsamples = 200

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
k = KernelPerceptron(numsamples)
#k.fit(X,Y)
#k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=True)

bk = BudgetKernelPerceptron(beta, N, numsamples)
bk.fit(X, Y)
bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=True)
