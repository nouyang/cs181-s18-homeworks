Note: using virtualenv

source venv/bin/activate

Requires sudo apt-get install python-dev
numpy
matplotlib



======
# 
======
import sklearn 
diabetes = sklearn.datasets.load_diabetes()

then you will get

AttributeError: module 'sklearn' has no attribute 'datasets'

This is a highly misleading error message, because sklearn does have a
subpackage called datasets - you just need to import it explicitly

import sklearn.datasets 
diabetes = sklearn.datasets.load_diabetes()

In [9]:  import sklearn.utils
In [11]: sklearn.utils.extmath.softmax

======
# 
======
blah

In [37]: help(np.reshape)


======
# 
======
blah

import sklearn.utils  ## for debugging  sklearn.utils.extmath.softmax
from sklearn.linear_model import LogisticRegression ## for debugging

    # clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)
    # clf_l2_LR.fit(X, y)

    # X, y = digits.data, digits.target
    # X = StandardScaler().fit_transform(X)

    # y = (y > 4).astype(np.int) # two classes: 0-4 vs 5-9

    # coef_l2_LR = clf_l2_LR.coef_.ravel()
    #     print("score with L2 penalty: %.4f" % clf_l2_LR.score(X, y))

# http://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic.html


http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


======
# 
======
blah

https://songhuiming.github.io/pages/2017/05/13/gradient-descent-in-solving-linear-regression-and-logistic-regression/

======
# 
======

Update: Clarifications on indicators, gradient, error, loss, and likelihood

Indicators -- just a mathematical notation trick.
Say to calculate some scalar metric, we only want to sum up the (error) from the
true values. 

Think about how, in terms of measuring how well our algorithm is
fitting the data, we only care about the predictions it is making. So we only
care about the error between the class we estimated and the real class. We don't
care about our other estimated errors.

However it is complicated to say "we would like to sum up the errors across the
datapoints, but only the error with our estimate for the true class, for each
datapoint".

Instead we are able to say the same thing we usually do, using sum(over
N)sum(over K), and use an indicator function (in bishop 4.2.2, denoted by `t_n`,
in the section notes, denoted by the big I) which evaluates to 0 for everything
except the "true class" as one-hot encoded from our training data, in which case
it is 1.

---

However, when figuring out *how* to improve the loss, that is when figuring out
the direction of steepest loss descent, we take the derivative of the loss. That
is we find the direction in which the loss changes the fastest.

The *negative log likelihood function* is what we define as our loss function.
L(w) = -ln(likelihood(w)) = -ll(w)

Note that in this case loss function *does not* equal likelihood function!
(We could use the likelihood function directly as our loss function, but it
simpilifies the math to use negative log likelihood instead).

It so happens that in the gradient of the likelihood includes a sort of `y_hat -
y` that looks like an error term. However --

It is incorrect to call *anything* in the gradient an "error" term. Though
something similar to an error term might appear, it would be very misleading.
Errors apply to the loss function. In this case we are picking a direction, so
the sign of the `y_hat - y` term does not indicate how large or small our error
is, it is just feeding into a decision about which direction to step in.

======
# 
======
blah


======
# 
======
blah


======
# 
======
blah


======
# 
======
blah


