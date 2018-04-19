# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
import time
from Perceptron import Perceptron
from numpy.core.umath_tests import inner1d
from matplotlib.backends.backend_pdf import PdfPages


# Implement this class
class KernelPerceptron(Perceptron):
    def __init__(self, numsamples):
        self.numsamples = numsamples
        self.alphas =  {} #of len(x)
        self.b = 0.0
        self.SV_indices = []
        self.X = []
        self.Y = []
        self.timefit = 0.0
        self.timepredict = 0.0

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
        self.timefit = t_elapsed
        print("Elapsed time for fitting ", X.shape, " datapoints: ", t_elapsed, " seconds." )

    def predict(self, X):
        t_start = time.time()
        print("=== Predicting ===")
        #print('support vector indices', self.SV_indices)
        #print('\nalphas', self.alphas)

        y_hats = []
        for xt in X:
            boo =  [self.alphas[sv_i]*np.dot(xt,self.X[sv_i]) for sv_i in self.SV_indices] 
            y_hat = np.sum(boo)
            if y_hat == 0:
                y_hat = 1
            y_hats.append(y_hat)
        y_hats = np.array(y_hats)
        y_hats = np.sign(y_hats)

        at_elapsed = time.time() - t_start
        self.timepredict = at_elapsed
        print("Elapsed time for predicting", X.shape, " datapoints: ", at_elapsed, "seconds.")
        print("samples %d, data shape %d" % (self.numsamples, self.X.shape[0]))
        return y_hats 


# Implement this class
class BudgetKernelPerceptron(Perceptron):
    def __init__(self, beta, N, numsamples):
        self.beta = beta
        self.N = N
        self.numsamples = numsamples
        self.X = X
        self.Y = Y
        self.timefit = 0.0
        self.timepredict = 0.0


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
                    del(SV_indices[arg_maxmarg])
        self.alphas = alphas
        self.SV_indices = SV_indices
        t_elapsed = time.time() - t_start
        self.timefit = t_elapsed
        print("Elapsed time for fitting ", X.shape, " datapoints: ", t_elapsed, " seconds." )
        print("N %d, beta %d, samples %d, data shape %d" % (self.N, self.beta, self.numsamples,
            self.X.shape[0]))

    def predict(self, X):
        t_start = time.time()
        print("=== Predicting ===")

        y_hats = []
        for xt in X:
            boo =  [self.alphas[sv_i]*np.dot(xt,self.X[sv_i]) for sv_i in self.SV_indices] 
            y_hat = np.sum(boo)
            if y_hat == 0: #dealing with the meshgrid has zeros at the beginning
                y_hat = 1
            y_hats.append(y_hat)
        y_hats = np.array(y_hats)
        y_hats = np.sign(y_hats)

        t_elapsed = time.time() - t_start
        self.timepredict = t_elapsed
        print("Elapsed time for predicting", X.shape, " datapoints: ", t_elapsed, "seconds.")
        return y_hats

# Do not change these three lines.
#data = np.loadtxt("data.csv", delimiter=',')
#X = data[:, :2]
#Y = data[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
numsamples = 20000
beta = 0 #budget
N = 100 #budget

n_list = [50, 100,200,500, 1000]
beta_list = [-5, -1, 0, 1, 5]
#kernel_file_name = 'k.png'
#budget_kernel_file_name = 'bk.png'


# # collect data
# plt.plot(nsamples, time)
# plt.plot(nsamples, accuracy)
# nsamples_list = [200,100,1000,20000]
# for num in nsamples_list:
    # k = KernelPerceptron(num)
    # k.fit(X,Y) 
    # astr = "Kernel Perceptron, # datapts=%d with %d samples, \nFit: %.02f sec" % \
            # (np.array(X).shape[0], num, k.timefit)
    # # astr = "Kernel Perceptron, # datapts=%d with %d samples, \nFit: %.02f sec, Predict: %.02f secs" % \
            # # (np.array(X).shape[0], num, k.timefit, k.timepredict)
    # print('predict time', k.timepredict)
    # print('fit time', k.timefit)
    # plt1 = k.visualize(kernel_file_name, width=0, show_charts=False, save_fig=False,\
            # include_points=True, text=astr)
    # plt1.savefig('kernel_%dsamples.png' % (num))
# #pp.savefig(aplot)



# for beta in beta_list:
    # bk = BudgetKernelPerceptron(beta, N, numsamples)
    # bk.fit(X, Y)
    # print('predict time', bk.timepredict)
    # print('fit time', bk.timefit)
    # bstr = "Budget Kernel Perceptron, \n%d datapoints with %d samples, %d N, %d beta \nFit: %.02f secs" % \
            # (np.array(X).shape[0], numsamples, N, beta, bk.timefit)
# #    plt2 = bk.visualize(budget_kernel_file_name, width=0, show_charts=False, save_fig=False,
# #            include_points=True, text=bstr)
    # #plt2.savefig('budgetkernel_%dsamples_%dN_%dbeta.png' % (numsamples, N, beta))

##############
# Calculate accuracy
##############
pts = 100
nsamples_list = np.asarray(np.arange(100,25000, (25000-100)/pts), dtype=int)
print(nsamples_list)
times = []
accuracies = []

data = np.loadtxt("data.csv", delimiter=',')
cval_data = np.loadtxt("val.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]
crossvalX = cval_data[:, :2]
crossvalY = cval_data[:, 2]

for num in nsamples_list:
    k = KernelPerceptron(num)
    k.fit(X,Y) 
    times.append(k.timefit)
    yhats = k.predict(crossvalX)
    test = [(y*y_hat>0) for y, y_hat in zip(crossvalY, yhats)]
    print('test shape',np.array(test).shape)
    print('num correct', sum(test))
    print('totaly sampes', len(Y))
    accuracy = sum([(y*y_hat>0) for y, y_hat in zip(crossvalY, yhats)]) / len(crossvalY)
    print('accuracy', accuracy)
    accuracies.append(accuracy)
#pp.savefig(aplot)
print(nsamples_list)
print(times)
print(accuracies)

fig, ax1 = plt.subplots()
ax1.plot(nsamples_list, times, 'b-')
ax1.set_ylabel("time (secs)")
ax1.yaxis.label.set_color("blue")
ax2 = ax1.twinx()
ax2.plot(nsamples_list, accuracies, 'r-')
ax2.yaxis.label.set_color("red")
ax2.set_ylabel("accuracy (#correct / total#)")
fig = plt.gcf()
fig.figsize=(5,5)
fig.dpi=100
bstr = "Kernel Perceptron, \n%d datapoints Fit: %.02f secs, accuracy across %d crossval points" % \
(np.array(X).shape[0], k.timefit, len(crossvalY))
plt.title(bstr)
plt.xlabel("# samples")
fig.tight_layout()
plt.show()



##############
# Calculate accuracy for BK, varying beta
##############
pts = 100
nsamples_list = np.asarray(np.arange(100,25000, (25000-100)/pts), dtype=int)
print(nsamples_list)
times = []
accuracies = []

data = np.loadtxt("data.csv", delimiter=',')
cval_data = np.loadtxt("val.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]
crossvalX = cval_data[:, :2]
crossvalY = cval_data[:, 2]

for num in nsamples_list:
    k = KernelPerceptron(num)
    k.fit(X,Y) 
    times.append(k.timefit)
    yhats = k.predict(crossvalX)
    test = [(y*y_hat>0) for y, y_hat in zip(crossvalY, yhats)]
    print('test shape',np.array(test).shape)
    print('num correct', sum(test))
    print('totaly sampes', len(Y))
    accuracy = sum([(y*y_hat>0) for y, y_hat in zip(crossvalY, yhats)]) / len(crossvalY)
    print('accuracy', accuracy)
    accuracies.append(accuracy)
#pp.savefig(aplot)
print(nsamples_list)
print(times)
print(accuracies)

fig, ax1 = plt.subplots()
ax1.plot(nsamples_list, times, 'b-')
ax1.set_ylabel("time (secs)")
ax1.yaxis.label.set_color("blue")
ax2 = ax1.twinx()
ax2.plot(nsamples_list, accuracies, 'r-')
ax2.yaxis.label.set_color("red")
ax2.set_ylabel("accuracy (#correct / total#)")
fig = plt.gcf()
fig.figsize=(5,5)
fig.dpi=100
bstr = "Kernel Perceptron, \n%d datapoints Fit: %.02f secs, accuracy across %d crossval points" % \
(np.array(X).shape[0], k.timefit, len(crossvalY))
plt.title(bstr)
plt.xlabel("# samples")
fig.tight_layout()
plt.show()

##############
# Calculate accuracy for bk, varying N
##############
pts = 100
nsamples_list = np.asarray(np.arange(100,25000, (25000-100)/pts), dtype=int)
print(nsamples_list)
times = []
accuracies = []

data = np.loadtxt("data.csv", delimiter=',')
cval_data = np.loadtxt("val.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]
crossvalX = cval_data[:, :2]
crossvalY = cval_data[:, 2]

for num in nsamples_list:
    k = KernelPerceptron(num)
    k.fit(X,Y) 
    times.append(k.timefit)
    yhats = k.predict(crossvalX)
    test = [(y*y_hat>0) for y, y_hat in zip(crossvalY, yhats)]
    print('test shape',np.array(test).shape)
    print('num correct', sum(test))
    print('totaly sampes', len(Y))
    accuracy = sum([(y*y_hat>0) for y, y_hat in zip(crossvalY, yhats)]) / len(crossvalY)
    print('accuracy', accuracy)
    accuracies.append(accuracy)
#pp.savefig(aplot)
print(nsamples_list)
print(times)
print(accuracies)

fig, ax1 = plt.subplots()
ax1.plot(nsamples_list, times, 'b-')
ax1.set_ylabel("time (secs)")
ax1.yaxis.label.set_color("blue")
ax2 = ax1.twinx()
ax2.plot(nsamples_list, accuracies, 'r-')
ax2.yaxis.label.set_color("red")
ax2.set_ylabel("accuracy (#correct / total#)")
fig = plt.gcf()
fig.figsize=(5,5)
fig.dpi=100
bstr = "Kernel Perceptron, \n%d datapoints Fit: %.02f secs, accuracy across %d crossval points" % \
(np.array(X).shape[0], k.timefit, len(crossvalY))
plt.title(bstr)
plt.xlabel("# samples")
fig.tight_layout()
plt.show()


##############
# Calculate accuracy for bk, varying num samples
##############
pts = 100
nsamples_list = np.asarray(np.arange(100,25000, (25000-100)/pts), dtype=int)
print(nsamples_list)
times = []
accuracies = []

data = np.loadtxt("data.csv", delimiter=',')
cval_data = np.loadtxt("val.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]
crossvalX = cval_data[:, :2]
crossvalY = cval_data[:, 2]

for num in nsamples_list:
    k = KernelPerceptron(num)
    k.fit(X,Y) 
    times.append(k.timefit)
    yhats = k.predict(crossvalX)
    test = [(y*y_hat>0) for y, y_hat in zip(crossvalY, yhats)]
    print('test shape',np.array(test).shape)
    print('num correct', sum(test))
    print('totaly sampes', len(Y))
    accuracy = sum([(y*y_hat>0) for y, y_hat in zip(crossvalY, yhats)]) / len(crossvalY)
    print('accuracy', accuracy)
    accuracies.append(accuracy)
#pp.savefig(aplot)
print(nsamples_list)
print(times)
print(accuracies)

fig, ax1 = plt.subplots()
ax1.plot(nsamples_list, times, 'b-')
ax1.set_ylabel("time (secs)")
ax1.yaxis.label.set_color("blue")
ax2 = ax1.twinx()
ax2.plot(nsamples_list, accuracies, 'r-')
ax2.yaxis.label.set_color("red")
ax2.set_ylabel("accuracy (#correct / total#)")
fig = plt.gcf()
fig.figsize=(5,5)
fig.dpi=100
bstr = "Kernel Perceptron, \n%d datapoints Fit: %.02f secs, accuracy across %d crossval points" % \
(np.array(X).shape[0], k.timefit, len(crossvalY))
plt.title(bstr)
plt.xlabel("# samples")
fig.tight_layout()
plt.show()




            # plt.xlabel('X dimension 1\n Red is class 1, Green is class -1')
            # plt.ylabel('X dimension 2')
            # plt.title(text)

# for beta in beta_list:
    # bk = BudgetKernelPerceptron(beta, N, numsamples)
    # bk.fit(X, Y)
    # print('predict time', bk.timepredict)
    # print('fit time', bk.timefit)
    # bstr = "Budget Kernel Perceptron, \n%d datapoints with %d samples, %d N, %d beta \nFit: %.02f secs" % \
            # (np.array(X).shape[0], numsamples, N, beta, bk.timefit)
# #    plt2 = bk.visualize(budget_kernel_file_name, width=0, show_charts=False, save_fig=False,
# #            include_points=True, text=bstr)
    # #plt2.savefig('budgetkernel_%dsamples_%dN_%dbeta.png' % (numsamples, N, beta))

##############
