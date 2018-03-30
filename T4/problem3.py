# CS 181, Spring 2017
# Homework 4: Clustering
# Name:
# Email:

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class KMeans(object):
    # K is the K in KMeans
    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    # EDIT: X is flattened to N.(28^2)
    def __init__(self, K):
        self.K = K
        self.dim = -1
        self.X = []
        self.numImages = -1
    #np.random.seed(314159)

    def dist(self, a,b):
        return np.linalg.norm(a-b) #l2 norm

    def closestCenters(self, imgs, centers):
        #print(centers)
        # we shall have something like [0,4,3,5,1,0,0] for each point, their cluster
        imgAssignments = []
        for img in imgs: 
            #print('img\n', img)
            minDist, k = 99999999, -999
            for kIndex in range(self.K):
                center = centers[kIndex]
                distortion = self.dist(img, center)
                if distortion < minDist:
                    minDist = distortion
                    k = kIndex
            imgAssignments.append(k)
        return imgAssignments

    # we have an array centers = [[gray1, gray2, ... 28^2] ... K]], for mean image of the K clusters
    def newCenters(self, imgs, clusterAssignments):
        #clusters = np.zeros(self.X.shape[0], self.dim**2)
        centers = []
        print('clusterAssignments\n', clusterAssignments)
        for k in range(self.K): #retrieve imgs that are for each cluster
            cluster = np.array([imgs[i] for i in range(self.numImages) \
                                if clusterAssignments[i]==k])
            print('k=', k, 'cluster\n', cluster)
            centers.append(np.mean(cluster, axis=0))
        print('centers\n', centers)
        return np.array(centers)

    def fit(self, X):
        self.dim = X.shape[1]
        K = self.K
        self.numImages = X.shape[0]
        self.X = X.reshape(self.numImages, -1) #flatten into 2D N.(28*28) array
        #currCenters = [np.random.rand(K) for k in range(self.K)]
        currCenters = np.random.randint(0,10,(self.numImages, self.dim, self.dim))
        print('hi')
        print(currCenters)
        #print(currCenters)
        #print(currCenters.shape)
        numIters = 5
        for i in range(numIters):
            print('now assigning points to clusters!!')
            clusterAssignments = self.closestCenters(X, currCenters)
            print('now calculating new centers!!')
            currCenters = self.newCenters(X, clusterAssignments)
        return clusterAssignments, currCenters

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        pass

    # This should return the arrays for D images from each cluster that are representative of the clusters.
    def get_representative_images(self, D):
        pass

    # img_array should be a 2D (square) numpy array.
    # Note, you are welcome to change this function (including its arguments and return values) to suit your needs. 
    # However, we do ask that any images in your writeup be grayscale images, just as in this example.
    def create_image_from_array(self, img_array):
        plt.figure()
        plt.imshow(img_array, cmap='Greys_r')
        plt.show()
        return

# This line loads the images for you. Don't change it! 
pics = np.load("images.npy", allow_pickle=False)
#print(pics[0])

# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, 
# though you may add more public methods for things like the visualization if you want.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.
K = 10
# what is useKMeansPP?
#https://www.rdocumentation.org/packages/pracma/versions/1.5.5/topics/kmeanspp
#KMeansClassifier = KMeans(K=10, useKMeansPP=False)
imgs = np.array([[num]*9 for num in range(5)]).reshape(5,3,3)

KMeansClassifier = KMeans(K=2)
KMeansClassifier.fit(imgs)
#KMeansClassifier = KMeans(K=10)
#KMeansClassifier.fit(pics)

#KMeansClassifier.create_image_from_array(pics[1])




#  grep -r seed --exclude-dir=venv
