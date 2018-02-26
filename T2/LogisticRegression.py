import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as c
from scipy.misc import logsumexp



# Bishop, chapter 4.3.4 Multilass logistic regression
# Bishop, chapter 5.2.4 Gradient descent optimization
# Bishop, page 10 Regularization 

# Logistic Regression Equation
# When differentiating loss w.r.t. the weights for class j, we take each
# datapoint x_i and add the following expression  ( which is the estimated
# probability of that datapoint x_i  being in class j, minus datapoint x_i's true
# label (either 0 or 1) for being in class j), which results in a scalar. Then
# we scape datapoint x_i by that scalar.
# This results in vector 
class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
        self.weights = np.array(-999) 
        self.iters = 50
    
    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None


    def __softmax(self, array_input):
        # Input:
        # - array of score for each class Dims: n.k
        # Output:
        # - vector of normalized scores, normalized by row  Dims: n.k
        exp_data =  np.exp(array_input)
        rowsum = np.sum(exp_data, axis=1)
        array_softmax = exp_data.T/rowsum
        array_softmax = array_softmax.T
        return array_softmax 

    def __grad_desc(self, x_input, w_input, true_c):
        print("--------------------------------------") #running gradient descent")
        # Input:
        # -- w - vector of weights k.d
        # -- x - array of parameters per datapoint -- n.d
        # Output:
        # - w_new - vector of w' = w - eta*grad(w) - l2 regularization -- k.d
        # - loss - ???

        ### TODO PULL OUT --
        x = np.copy(x_input)
        weights = np.copy(w_input)
        trueclass = np.copy(true_c)

        n = x.shape[0] # number of datapoints
        d = x.shape[1] # number of features per datapoint
        k = 3
        ### -- 
        C_hot = np.eye(k)[np.array(true_c).reshape(-1)] #one-hot. hattip to internet

        wdotx = np.dot(x, weights.T) # n.k
        #print(x[1:2,:])
        #print(weights.T)
        #print(wdotx[1:2,])
        zscores = 1.0 / (1.0 + np.exp(-wdotx)) # sigmoid fxn Dims: n.k
        softscores = self.__softmax(zscores) # n.k

        diffs = softscores - C_hot # n.k Question: Do we not want the absolute value of y_est - y? 
        # A: no, this is a value for calculating the slope of the loss, not the loss itself!!
        # WRONG: class_errs = np.sum(errs, axis=0) # 1.k 
        for j in range(k): 
            gradj = np.zeros(d) #1.d
            for i in range(n):  #rows
                xi = x[i,:] #vector  1.d
                gradj += diffs[i][j] * xi #scalar 1.1 * vector 1.d
            reg = self.lambda_parameter * np.dot(weights[j,:], weights[j,:]) # ?  
            reg = 0
            weights[j,:] = weights[j,:] - (gradj*self.eta + reg) #update step
            
        est_hot_weights =  np.array([weights[true,:] for true in true_c]) #weights for data labelled true
        y_est = np.array([ np.dot(foow, foox) for foow,foox in zip(est_hot_weights,self.X)])
        # y_est = 
        # est_weights = C_hot * weights
        #print(y_est)
        est_sum = sum(y_est)
        err = n - est_sum
        #print("error: ",err)
        return weights, err 

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
        
        weights = np.random.rand( k, d) #init to random
        for z in range(self.iters):
            weights, loss = self.__grad_desc(self.X, weights, self.C) 
            #print(weights, loss)
            print("iter: ", z, "loss: ", loss)

        self.weights = weights

        #loss = for the true class k 1 - np.dot(x[n],self.weights[k,:])
        return #predicted weights

    # TODO
    # Given new set of x's, and our trained model, predict ys (one of 3 classes)
    # y is {0,1,2}
    def predict(self, X_to_predict):
        Y = []
        for rowx in X_to_predict:
            predicts = []
            for rowweight in self.weights:
                predict = np.dot(rowx, rowweight)
                predicts.append(predict)
            class_x = np.argmax(predicts)
            Y.append(class_x)
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
        cMapScatter = c.ListedColormap(['r','b','g'])
        #cMapScatter = c.ListedColormap(['m','c','y'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.C, cmap=cMapScatter,
                edgecolors='k')
        plt.savefig(output_file)
        if show_charts:
            plt.show(losloslossss)


# This problem was very confusing due to the lack of definitions between
# loss, error, gradient of likelihood, log-likelihood 
# Also the indices were very confusing. What are l, j, i, in the gradient
# And whether or not the updates were per-weight or per-class etc.
# Finally, the difference between the loss function and the gradient update
# function

# Update: see ./README.md for answers to above
