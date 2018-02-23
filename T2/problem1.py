#!/bin/python3
import numpy as np
import matplotlib as plt
#print("hellow world")

# Beta Bernoulli model

data = [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0]
print("len data", len(data))

# theta_MLE
# This uses only the knowledge of the possibile "coins"
# (in this case the likelihood, as modeled by the Bernolli model)
# and the observed data so far



def MLE(num0, num1):
    if (num0 == 0 and num1 == 0):
        raise ValueError("Both num0 and num1 are zero")
    else:
        return float(num1) / (num0 + num1)

def MAP(alpha, beta, num0, num1):
    theta = float(alpha+num1-1) / (alpha+beta+num1+num0-2)
    return theta

def postpred(alpha, beta, num0, num1):
    pred = float(alpha+num1) / (alpha+beta+num1+num0)
    return pred
steps = [0,4,8,12,16]
 
# theta_MAP
# This use our prior, as modeled by the Beta distribution 
# and the observed data so far
