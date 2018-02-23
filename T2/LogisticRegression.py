import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import logsumexp


import sklearn.utils  ## for debugging  sklearn.utils.extmath.softmax

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
    
    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def softmax(array_input):
        X = np.copy(X)
        max_prob = np.max(X, axis=1).reshape((-1, 1))
        X -= max_prob
        np.exp(X, X)
        sum_prob = np.sum(X, axis=1).reshape((-1, 1))
        X /= sum_prob
        return X



    # TODO
    # Run this before predict to produce a function we can compute on new x
    # values
    def fit(self, X, C):
        self.X = X
        self.C = C
        return

    # TODO
    # Given new set of x's, and our trained model, predict ys
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
