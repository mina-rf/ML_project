from mnist import MNIST
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


def purity_score(clusters, classes):

    A = np.c_[(clusters, classes)]
    n_accurate = 0.
    for j in np.unique(A[:, 0]):
        z = A[A[:, 0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]



mndata = MNIST('data')
tr_imgs , tr_lbs = mndata.load_training()
test_imgs , test_lbs = mndata.load_testing()

new_tr_lbs = [item for item in tr_lbs]
new_test_lbs = [item for item in test_lbs]

new_train = []
for img in tr_imgs :
    pixels = np.array(img, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    img_hog = hog(pixels)
    new_train.append(img_hog)

new_test = []
for img in test_imgs :
    pixels = np.array(img, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    img_hog = hog(pixels)
    new_test.append(img_hog)


pca = PCA(n_components=30)
new_train = pca.fit_transform(new_train)
new_test = pca.fit_transform(new_test)


# gmm = GaussianMixture(n_components=10 , n_init=10).fit(new_train)
# pre_lbs_gmm = gmm.predict(new_test)

kmeans = KMeans(n_clusters=10, random_state=0).fit(new_train)
pre_lbs_kmeans = kmeans.predict(new_test)



print('purity ',purity_score(pre_lbs_kmeans,new_test_lbs))



