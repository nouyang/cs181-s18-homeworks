#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
#print("hello world")


def MLE(num0, num1):
    if (num0 == 0 and num1 == 0):
        raise ValueError("Both num0 and num1 are zero")
    else:
        return float(num1) / (num0 + num1)

def MAP(alpha, beta, num0, num1):
    theta = float(alpha+num1-1) / (alpha+beta+num0+num1-2)
    return theta

def postpred(alpha, beta, num0, num1):
    pred = float(alpha+num1) / (alpha+beta+num1+num0)
    return pred

# our updated priors
def postbeta(alpha, beta, num0, num1):
    updated = (num1 + alpha,  num0 + beta)
    return updated

# Beta Bernoulli model

data = [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0]
num0s = []
num1s = []

for i in range(len(data)):
    #num1s.append(sum(data[:i+1])) #careful with indexing
    num0s.append(data[:i+1].count(0)) #could be more clever and use ~ i-count(1s)
    num1s.append(data[:i+1].count(1))

#print("len data", len(data))

alpha_init = 4
beta_init = 2

counts = zip(num0s, num1s)

MLEs = [MLE(num0, num1) for num0, num1 in counts]
MAPs= [MAP(alpha_init, beta_init, num0, num1) for num0, num1 in counts]
postpreds = [postpred(alpha_init, beta_init, num0, num1) for num0, num1 in counts]
postbetas = [postbeta(alpha_init, beta_init, num0, num1) for num0, num1 in counts]
postbetas.insert(0, postbeta(4,2,0,0))

print("MLEs ", MLEs)
print("MAPs ", MAPs)
print("postpreds", postpreds)


# plot MLE vs samples

xs = range(len(MLEs))

plt.scatter(xs, MLEs, label="MLEs")
plt.scatter(xs, MAPs, label="MAPs")
plt.scatter(xs, postpreds, marker='x',label="Postpreds")


plt.xlabel('Number of samples')
plt.ylabel('Thetas')
plt.title('Problem 1.1 - posterior predictive distribution, MAPs, MLEs')
# Why is it so hard to add a caption in pyplot??
#caption = '''Using Beta-Bernoulli model, with Beta(4,2), and data = [0, 0, 1, 1,
#0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0]'''
#plt.figtext(0.1, 0.1, caption)
plt.legend()
############# RUN THIS ################
#plt.show()

x = np.linspace(0, 1.0, 100)
steps = [0,4,8,12,16]
plt.figure()

for step in steps:
    betaparam = postbetas[step] 
    text = "Ex. "+ str(step) + ", beta" + str(betaparam)
    plt.plot(x, beta.pdf(x, *betaparam), label= text)

plt.legend()
plt.ylabel('PDF')
plt.title('Problem 1.2 - Posterior Distribution for 0,4,8,12,16 Examples')
plt.show()


#######################################
####### Problem 1.2 ###################









#### Notes to self 
# theta_MLE
# This uses only the knowledge of the possibile "coins"
# (in this case the likelihood, as modeled by the Bernolli model)
# and the observed data so far

# theta_MAP
# This use our prior, as modeled by the Beta distribution 
# and the observed data so far
