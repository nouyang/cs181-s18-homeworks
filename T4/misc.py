import numpy as np
import matplotlib.pyplot as plt
const = 6000*28**2

objs = np.array([2.970801283825729,
                 2.1593113092850316,
                 2.053740173112835,
                 2.028430170271818,
                 2.020434618936546,
                 2.014945309442909,
                 2.009952545073193,
                 2.0058719436530668,
                 2.0020712697939245,
                 1.9991800240967197,
                 1.9956847391084258,
                 1.9925334013885792,
                 1.9902212840037334,
                 1.9887275711185066,
                 1.9879460378803453,
                 1.9874940357430189,
                 1.9872573609899558,
                 1.9871233588398067,
                 1.9869498237054917,
                 1.9868501233662605,
                 1.9868079718580767,
                 1.986795990298887,
                 1.9867798515068982,
                 1.986773534909377,
                 1.986773534909377])



xs = np.arange(objs.shape[0])
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(xs, objs*const, marker='x')
plt.ylim(ymin=0)
plt.title('MNIST Kmeans Objective, K=10, converged at iter=24\n')
plt.xlabel('Iterations')
plt.ylabel('Objective (l2 norm)')

# Aborted attempt to label each point to show decreasing cost
# #labels = ['%0.02f' % obj for obj in objs]
# for xy in zip(xs, objs):                                       # <--
    # ax.annotate('(%.02f, %.02f)' % xy, xy=xy, xytext=(10,20), textcoords='offset points')
plt.show()
