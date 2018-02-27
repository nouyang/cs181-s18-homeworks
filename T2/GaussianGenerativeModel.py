from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class GaussianGenerativeModel:
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        n = x.shape[0] # number of datapoints
        d = x.shape[1] # number of features per datapoint
        k = 3
        ### -- 

        C_hot = np.eye(k)[np.array(true_c).reshape(-1)] #one-hot. hattip to internet
        N_k = np.sum(C_hot, axis=0) # get num pts per class, sum col to  
        priors = N_k / n
        temp =  # wtf why can i not stack on an array to another vertically

        # okay we have... X's ... and we only want the values to some of them...
        # each row of X is indexed by k...

        temp = np.hstack((np.array([Y]).T, X)) # ugh shenangians to concatenate a 1d column
        class_xsums = np.array([])
        for j in range(k):
            foo = np.ones(k)
            for i in range(n):
                if Y[i] == j:
                    foo = np.concatenate((foo,X[i]))
#        class_xsums = np.sum(class_xsums, axis=0) 
#        means = class_xsums / N_k



        a = np.random.rand(N,N)
        b = np.zeros((N,N+1))
        b[:,:-1] = a



    return

    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        for 
        Y = []
        for x in X_to_predict:
            Y.append(0)

        return np.array(Y)

    # Do not modify this method!
    def visualize(self, output_file, width=3, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
            y_max, .005))

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
        plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
