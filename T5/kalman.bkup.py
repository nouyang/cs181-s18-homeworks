
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  15 Apr

@author: nrw 
"""

import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('./kf-data.csv' )
print(list(df))
print(df.head())

# Get x and y
z = df['z'].values.tolist()
x = df['x'].values.tolist()
t = df['Unnamed: 0'].values.tolist()
x[10] = 10.2

#multiplying two gaussians
def gauss_update(mean1, var1, mean2, var2):
    new_mean = (var2*mean1 + var1*mean2) / (var2 + var1)
    new_var = 1 / ((1/var1)+ (1/var2))
    return new_mean, new_var


# here likelihood = our measurement
def update(likelihood_mean, likelihood_var, prior_mean, prior_var):
    posterior_mean, posterior_var = gauss_update(
        likelihood_mean, likelihood_var, prior_mean, prior_var)
    return (posterior_mean, posterior_var)

def predict(prior_mean, prior_var, motion_mean, motion_var):
    post_mean = prior_mean + motion_mean
    post_var = prior_var + motion_var
    return post_mean, post_var


# aka z moves by kind of dithering around the same point (epsilon)
# and x is a noisy measurement of z (gamma)
mean_eps = 0
var_eps = 0.05**2

mean_y = 0
var_y = 1**2
meas_var = var_y #constant

prior_mean = 5
prior_var = 1**2


zs = []
zs_var = []
errs = []
for step in t:
    # update with new measurement ll
    meas = x[step]
    meas_mean = meas
# our current estimate is mean, var
    mean, var = update(meas_mean, meas_var, prior_mean, prior_var) #measurement
    mean, var = predict(mean, var, mean_eps, var_eps) #motion
# let our current estimate by the prior for the next step
    prior_mean, prior_var = mean, var

    zs.append(mean)
    zs_var.append(var)

    print("measured %0.02f, curr est mean %0.02f, curr est var %0.02f, true_z %0.02f" % \
          (meas, mean, var, z[step]))

e = np.array(zs_var)*2
plt.errorbar(t, zs, e, label='est z',linewidth=1, marker='.', capsize=3)
#plt.plot(t, zs)
plt.plot(t, z, label='true z')
# plt.plot(t, x, label='measurement')
plt.legend()
plt.show()



# Hi,
# We are referring to the effect that 'incorrect' parameters would have. So, if the given parameters
# μϵ/γ/p, σϵ/γ/p are different from the original parameters that were used to generate the data, how
# would your predictions about the hidden state trajectory get affected as t goes from 1 to T.
# Hope this helps!

