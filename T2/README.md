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


======
# 
======
blah


======
# 
======
blah


