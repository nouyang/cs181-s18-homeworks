import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as c
from scipy.misc import logsumexp



# Bishop, chapter 4.3.4 Multilass logistic regression
# Bishop, chapter 5.2.4 Gradient descent optimization
class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
    
    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None


    def softmax(array_input):
        # Input
        # -- X, vector of score for each class Dims: 1.k
        # Output
        # -- vector of normalized scores for each class  Dims: 1.k
        scores = np.exp(array_input)
        sumscore = np.sum(scores)
        softmax_scores =  scores / sumscore
        return softmax_scores

    # TODO
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
        self.C = C # true class :

        n = X.shape[0]
        d = X.shape[1]
        k = max(C)

        # Initialize w estimates with 1s. Dims: 1.d
        # Since we know a priori the # of classes, we'll list 'em out for now
        # for clarity
        w1 = np.ones([1,d])  # 1st of k classes, weights for each parameter (we have d=2)
        w2 = np.ones([1,d]) 
        w3 = np.ones([1,d]) 

        Weights = np.vstack([w1,w2,w3]) # k.d
        # z score for each datapoint belonging to each of the classes. k.n 
        # matrix multiplication: dot product of each A.row by B.column 
        # thus, k.D x n.D 
        Scores = np.multiply(Weights,X.T)

        Scores =  
        score_k1 = np.dot(w1, X[:,0].T)  
        score_k2 = np.dot(w2, X.T)
        score_k3 = np.dot(w3, X.T)

        score_k1_param2
        score_k2 = np.dot(w2, X.T)
        score_k3 = np.dot(w3, X.T)

        scores = np.hstack([score_k1, score_k2, score_k3]) # 1.k, where K = numclasses
        softmax_scores = softmax(scores) # k.1, where axis=0 is [1...k] class per datapt

        class_err = (softmax_classes - C)  # for each class, sum up total error across datapts
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
