from mnist import MNIST
import numpy as np
from skimage.feature import hog , daisy
from sklearn import manifold
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import cv2
import time
import matplotlib.pyplot as plt

from gmm import GMM
from k_means import MyKmeans

start_time = time.time()


def purity_score(clusters, classes):

    A = np.c_[(clusters, classes)]
    n_accurate = 0.
    for j in np.unique(A[:, 0]):
        z = A[A[:, 0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]

def deskew(img):
    SZ= 20
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


mndata = MNIST('data')
tr_imgs , tr_lbs = mndata.load_training()
test_imgs , test_lbs = mndata.load_testing()

new_tr_lbs = [item for item in tr_lbs]
new_test_lbs = [item for item in test_lbs]

print(purity_score([2,1,2,3,2,2,3,1] , [1,1,2,3,2,2,3,1]))

new_train = []
for img in tr_imgs :
    pixels = np.array(img, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    pixels = deskew(pixels)
    pixels = np.pad(pixels, ((4,4),(4,4)), mode='constant')
    img_hog = hog(pixels,block_norm='L2-Hys')
    new_train.append(img_hog)

print('here')
new_test = []
for img in test_imgs :
    pixels = np.array(img, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    pixels = deskew(pixels)
    # print(np.shape(pixels))
    pixels = np.pad(pixels, ((4, 4), (4, 4)), mode='constant')
    img_hog = hog(pixels,block_norm='L2-Hys')
    # print(np.shape(img_hog))
    new_test.append(img_hog)
print(np.shape(new_train))

print('here2')

# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
# t0 = time()
# tsne.fit(new_train)
# X_tsne = tsne.transform(new_train)

pca = PCA(n_components=20)
fitted_pca = pca.fit(new_train)
new_train = fitted_pca.transform(new_train)
new_test = fitted_pca.transform(new_test)

print('here3')
# gmm = GaussianMixture(n_components=10).fit(tr_imgs)
# pre_lbs_gmm = gmm.predict(test_imgs)


# kmeans = MyKmeans(n_components=10)
# kmeans.fit(new_train)#KMeans(n_clusters=10).fit(X_tsne)
# pre_lbs_kmeans = kmeans.predict(new_test)

gmm = GMM(n_clusters=10)
gmm.fit(new_train)
pre_lbs_gmm = gmm.predict(new_test)

print('purity ',purity_score(pre_lbs_gmm,new_test_lbs))



print("--- %s seconds ---" % (time.time() - start_time))


