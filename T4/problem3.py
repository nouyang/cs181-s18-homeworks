# CS 181, Spring 2017
# Homework 4: Clustering
# Name:
# Email:

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
import random

class KMeans(object):
    # K is the K in KMeans
    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    # EDIT: X is flattened to N.(28^2)
    def __init__(self, K):
        self.K = K
        self.dim = -1
        self.X = []
        self.numImages = -1
        self.centers = []
        self.distortions = []
        self.ks = []
        #np.random.seed(314159)

    def dist(self, a,b):
        return np.linalg.norm(a-b) #l2 norm

    def closestCenters(self, imgs, centers):
        #imgAssignments = np.zeros((self.numImages, 1))
        imgAssignments = [] 
        distortions = []
        for img in imgs: 
            minDist, k = np.inf, np.inf
            for kIndex in range(self.K):
                center = centers[kIndex]
                distortion = self.dist(center, img)
                if distortion < minDist:
                    minDist = distortion
                    k = kIndex
            imgAssignments.append(k)
            distortions.append(minDist)
        self.distortions = distortions #a min distortion per img
        return imgAssignments

    # we have an array centers = [[gray1, gray2, ... 28^2] ... K]], for mean image of the K clusters
    def newCenters(self, imgs, clusterAssignments):
        centers = []
        print('clusterAssignments\n', clusterAssignments[0:10])
        for k in range(self.K): #retrieve imgs that are for each cluster
            cluster = np.array([imgs[i] for i in range(self.numImages) \
                                if clusterAssignments[i]==k])
            if cluster.size:
                centers.append(np.mean(cluster, axis=0))
            else:
                print('WARNING: Had to randomly reinitialize a center!')
                centers.append( np.random.randint(0,2,(self.dim, self.dim)))
        self.centers = centers
        print('CENTERS\n', [c.tolist() for c in centers])
        return np.array(centers)

    def fit(self, X, numIters):
        print('imgs', X)
        np.set_printoptions(precision=3)
        self.dim = X.shape[1]
        K = self.K
        self.numImages = X.shape[0]
        print('self.numImages', self.numImages)
        self.X = np.array(X)
        currCenters = np.random.randint(0,2,(K, self.dim, self.dim))
        print('K received: ', K, '\n Centers: ', currCenters)
        for i in range(numIters):
            print('~~~~~ i_th iteration', i, '~~~~~')
            clusterAssignments = self.closestCenters(X, currCenters)
            objective = np.sum(self.distortions)# / (self.numImages*28**2)
            currCenters = self.newCenters(X, clusterAssignments)
            print('--> objective ------->', objective)
        self.ks = np.array(clusterAssignments)
        return clusterAssignments, currCenters

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        #print('shape of centers', np.array(self.centers).shape)
        return self.centers


    # This should return the arrays for D images from each cluster that are representative of the clusters.
    def get_representative_images(self, D):
        allReps = []
        for k in range(self.K-1):
            reps = []
            onehot = self.distortions * (self.ks==k)
            classDistorts = [ (i,item) for i, item in enumerate(onehot)]
            classDistorts = [ i for i in classDistorts if i[1] != 0]
            mindists = sorted(classDistorts, key=lambda x: x[1]) #sort by distance, retaining index
            #assert len(mindists)!= 0
            if len(mindists) == 0:
                repindices = [0]*D
            else:
                #print('k',k, 'D', D,'len', len(mindists[:D]), 'len mindist', len(mindists))
                #print(np.array(mindists[:D])[0,:])
                repindices = np.asarray(np.array(mindists[:D])[0,:], dtype=int)
                #print(repindices)
            reps = self.X[repindices]
            allReps.append(reps)
        return allReps # D is ... two images for now


    # img_array should be a 2D (square) numpy array.
    # Note, you are welcome to change this function (including its arguments and return values) to suit your needs. 
    # However, we do ask that any images in your writeup be grayscale images, just as in this example.
    def create_image_from_array(self, img_array, filename='null'):
        fig = plt.figure()
        plt.imshow(img_array, cmap='Greys_r')
        #fig.savefig(filename)
        plt.show()
        return

# This line loads the images for you. Don't change it! 
pics = np.load("images.npy", allow_pickle=False)
#print(pics[0])

# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, 
# though you may add more public methods for things like the visualization if you want.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.
K = 2

# what is useKMeansPP? let us ignore it for now
#https://www.rdocumentation.org/packages/pracma/versions/1.5.5/topics/kmeanspp
#KMeansClassifier = KMeans(K=10, useKMeansPP=False)

numIters = 5

imgs = np.array([[num]*4 for num in range(2)]).reshape(2,2,2)
KMeansClassifier = KMeans(K=2)
KMeansClassifier.fit(imgs, numIters)

c =  np.array(KMeansClassifier.centers).reshape(2,2,2)
print(c)
print(len(c))

# KMeansClassifier = KMeans(K=10)
# KMeansClassifier.fit(pics, numIters)
# blah = KMeansClassifier.get_mean_images()
# print(blah[0].shape)
# fig = plt.subplots()
# for i, image in enumerate(blah):
    # plt.subplot(2,5, i+1)
    # plt.axis('off')
    # plt.imshow(image, cmap='Greys_r')
    # plt.title('Cluster: %i' % i)
# plt.tight_layout()
# plt.suptitle('MNIST Kmeans with %i iters and %i clusters' % (numIters, K))
# time = datetime.now().strftime('%H:%M:%S')
# fname = 'centroid_%i_iters_%i_clusters_' % (numIters, K) + time + '.png'
# plt.savefig(fname)
# plt.show()

# reps = KMeansClassifier.get_representative_images(2)

#print( np.array(reps[0].reshape(28,28)).ndim)
#arep = np.array(reps[0]).reshape(28,28)
#KMeansClassifier.create_image_from_array(reps[0][0])
#KMeansClassifier.create_image_from_array(reps[1][0])
#plt.create_image_from_array(rep[1][1])

# for k in range(K):
    # fname = 'meanimage_' + str(k) + '.png'



#  grep -r seed --exclude-dir=venv
