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
        self.means = [] 
        self.covars = [] 
        self.priors = []

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        n = X.shape[0] # number of datapoints
        d = X.shape[1] # number of features per datapoint
        k = 3
        ### -- 

        C_hot = np.eye(k)[np.array(self.Y).reshape(-1)] #one-hot. hattip to internet
        N_k = np.sum(C_hot, axis=0) # get num pts per class, sum col to  
        priors = N_k / n

        # okay we have... X's ... and we only want the values to some of them...
        # each row of X is indexed by k...

        #temp = np.hstack((np.array([true_c]).T, X)) # ugh shenangians to concatenate a 1d column
        # delete the other rows???

        print("began train")
        class_xs = []
        class_sums = []
        for j in range(k):
            xs = []
            indices = [i for i, x in enumerate(Y) if x != j]
            xs = np.delete(np.array(X), indices, axis=0)
            class_xs.append(xs)
            class_sums.append(sum(xs))
        class_sums = np.array(class_sums)
        means = np.array(class_sums.T / N_k).T
        diffs = [] #covar  per class
        for i in range(k):
            diff = class_xs[i] - means[i]
            coldiff = np.sum(np.square(diff))
            diffs.append(coldiff)
        covars = np.array(diffs) / n 
        self.means = means
        self.covars = covars
        self.priors = priors
        print("finished train")
        return

    def predict(self, X_to_predict):
        Y = []
        k = 3
        errs = []
        for x in X_to_predict:
            #print("now on x : ", x)
            xtemp =  []
            if self.isSharedCovariance:
                covar = np.sum(self.covars)
                for j in range(k):
                   logLL= np.log( self.priors[j]) \
                           + np.dot(x,self.means[j]) * covar **-1\
                           - 0.5 * np.sum(np.square(self.means[j])) * covar**-1
                   xtemp.append(logLL)
                errs.append(max(xtemp))
                class_x = np.argmax(xtemp)
                Y.append(class_x)
            else:
                for j in range(k):
                   logLL= np.log( self.priors[j]) - 0.5*np.log( np.abs(self.covars[j]))\
                        - 0.5 *(self.covars[j]**-1) * np.sum(np.square(x - self.means[j])) 
                   xtemp.append(logLL)
                class_x = np.argmax(xtemp)
                errs.append(max(xtemp))
                Y.append(class_x)
            #print(errs)


        plt.clf()
        plt.plot()
        print(errs)
        plt.plot(range(len(errs)), errs)

        plt.xlabel('Iterations')
        plt.ylabel('Losses')
        astring = "Problem 3 Gauss"  + str(self.isSharedCovariance)
        plt.title(astring)
        plt.draw()

        plt.pause(1) # <-------
        raw_input("<Hit Enter To Close>")
        plt.clf()
        plt.close()
        return np.array(Y)

    # Do not modify this method!
    def visualize(self, output_file, width=3, show_charts=False):
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
        plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap,
                edgecolors='k')

        plt.savefig(output_file)
        if show_charts:
            plt.show()
