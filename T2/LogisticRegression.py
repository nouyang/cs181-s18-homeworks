import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as c
from scipy.misc import logsumexp



# Bishop, chapter 4.3.4 Multilass logistic regression
# Bishop, chapter 5.2.4 Gradient descent optimization
# Bishop, page 10 Regularization 
class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
    
    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None


    def softmax(array_input):
        # Input:
        # - array of score for each class Dims: n.k
        # Output:
        # - vector of normalized scores, normalized by row  Dims: n.k
        exp_data =  np.exp(array_input)
        rowsum = np.sum(exp_data, axis=1)
        array_softmax = (exp_data/1.0) / sumscore # divide by 1.0 b/c of int division paranoia
        return array_softmax 
    def err

    def grad_desc(w, x):
        # Input:
        # -- w - vector of weights
        # -- x - array of parameters per datapoint
        # Output:
        # - w_new - vector of w' = w - eta*grad(w) - l2 regularization
        # - class_errs - vector of error for each class
            w_old = w_new
            w_reshaped = w.reshape([d, k]) # d.k
            wdotx = np.multiply(x, w_reshaped.T) # n.k
            zscores = 1.0 / (1.0 + exp(wdotx)) # sigmoid fxn Dims: n.k
            softscore = self.softmax(z_scores) # n.k
            errs = softscores - C # n.k Question: Do we not want the absolute value of y_est - y?
            class_errs = np.sum(errs, axis=0) # 1.k 
            regterm = (self.lamda*0.5) * w_old**2  # Question: old w or w + grad(w)? does it matter
            w_new = w_old - (self.eta*class_errs + regterm) # 1.w
        return w_new, sum(class_errs)

    # todo: document the math behind all of this matrix manipulation
    # Run this before predict to produce a function we can compute on new x
    # values
    def fit(self, X, C):
        # Input:
        # - matrix of data X: -- (n pts by d parameters Dims: N.D)
        # -- {reals}  (i.e. length in cm, weight in cm)
        # - column vector of true classes for each x (Dims: N.1)
        # -- {0 1 2} (i.e. apples oranges lemons)
        # Output:
        # - vector of weights w, one for each class N.D
        self.X = X # shape N.D
        self.C = C # true class -- convert to one-hot

        n = X.shape[0] # number of datapoints
        d = X.shape[1] # number of features per datapoint
        #k = max(C) # number of classes #Nah... what if our training set is missing a class entirely
        k = 3
        w = k*n # number of weights

        C_hot =np.eye(k)[np.array(C).reshape(-1)] #one-hot. hattip to internet
        
        w_init = ones([1, k*d]) # todo: probably should be random?

        # calc_loss = 

        # run gradient descent
        return #predicted weights

    # TODO
    # Given new set of x's, and our trained model, predict ys (one of 3 classes)
    # y is {0,1,2}
    def predict(self, X_to_predict):
        Y = []
        for x in X_to_predict:
            val = 0
            if x[1] > 4:
                val += 1
            if x[1] > 6:
                val += 1
            Y.append(val)
        return np.array(Y)

    # Done.
    def visualize(self, output_file, width=2, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
            y_max, .05))

        # Flatten the grid so the values match spec for self.predict
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        X_topredict = np.vstack((xx_flat,yy_flat)).T

        # Get the class predictions
        Y_hat = self.predict(X_topredict)
        Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))
        
        cMap = c.ListedColormap(['r','b','g'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.C, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()


# This problem was very confusing due to the lack of definitions between
# loss, error, gradient of likelihood, log-likelihood 
# Also the indices were very confusing. What are l, j, i, in the gradient
# And whether or not the updates were per-weight or per-class etc.
# Finally, the difference between the loss function and the gradient update
# function
