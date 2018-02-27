import pandas as pd
eta = .1
lambda_parameter = .001
df = pd.read_csv("fruit.csv")
X = df[['width', 'height']].values
Y = (df['fruit'] - 1).values

n = X.shape[0] # number of datapoints
d = X.shape[1] # number of features per datapoint
k = 3
### -- 


weights = np.random.rand( k, d) #init to random

def softmax(array_input):
    # Input:
    # - array of score for each class Dims: n.k
    # Output:
    # - vector of normalized scores, normalized by row  Dims: n.k
    exp_data =  np.exp(array_input)
    rowsum = np.sum(exp_data, axis=1)
    array_softmax = exp_data.T/rowsum
    array_softmax = array_softmax.T
    return array_softmax 


x = np.copy(X)
true_c= np.copy(Y)

n = x.shape[0] # number of datapoint
d = x.shape[1] # number of features per datapoint
k = 3
### -- 
C_hot = np.eye(k)[np.array(true_c).reshape(-1)] #one-hot. hattip to internet


def grad(w):
    wdotx = np.dot(x, weights.T) # n.k
    zscores = 1.0 / (1.0 + np.exp(-wdotx)) # sigmoid fxn Dims: n.k
    softscores = softmax(zscores) # n.k

    diffs = softscores - C_hot # n.k Question: Do we not want the absolute value of y_est - y? 
    for j in range(k): 
        gradj = np.zeros(d) #1.d
        for i in range(n):  #rows
            xi = x[i,:] #vector  1.d
            gradj += diffs[i][j] * xi #scalar 1.1 * vector 1.d
        reg = lambda_parameter * np.dot(weights[j,:], weights[j,:]) # ?  
        reg = 0
        weights[j,:] = weights[j,:] - (gradj*eta + reg) #update step

    est_hot_weights =  np.array([weights[true,:] for true in true_c]) #weights for data labelled true
    y_est = np.array([ np.dot(foow, foox) for foow,foox in zip(est_hot_weights,X)])
    est_sum = sum(y_est)
    err = n - est_sum
    return weights, err 


weights = np.random.rand( k, d) #init to random
print(weights)
foow, fooloss = grad(weights)
print(foow)
print(fooloss)

====================
for rowweight in self.weights:
    predict = np.dot(rowx, rowweight)
    predicts.append(predict)
    class_x = np.argmax(predicts)
    Y.append(class_x)

C_hot = np.eye(3)[np.array(Y).reshape(-1)] #one-hot. hattip to internet
N_k = np.sum(C_hot, axis=0) # get num pts per class, sum col to  

class_xsums = np.array([])
class_xs = np.ones([59,3])
class_xs = 
 
        # should be more clever and write withfilters but WHO CARES
for j in range(3):
    xs = []
    for i in range(59):
        if Y[i] == j:
            xs.append(X[i])
    class_xs.append(xs)

class_xsums = np.sum(class_xs, axis=0)  # array of 3 xsum's with 2 val each  = 6.1
means = class_xsums / N_k
