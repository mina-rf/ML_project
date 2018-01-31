import cv2
import numpy as np
from mnist import MNIST
from skimage.feature import hog
from sklearn import svm
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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


mndata = MNIST('data',return_type='numpy')
tr_imgs, tr_lbs = mndata.load_training()
test_imgs, test_lbs = mndata.load_testing()

print(tr_lbs)
new_tr_lbs = [item for item in tr_lbs]
new_test_label = [item for item in test_lbs]

new_train = []
for img in tr_imgs:
    pixels = np.array(img, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    pixels = deskew(pixels)
    # print(np.shape(pixels))
    pixels = np.pad(pixels, ((4, 4), (4, 4)), mode='constant')
    img_hog = hog(pixels, block_norm='L2-Hys')
    new_train.append(img_hog)

new_test = []
for img in test_imgs:
    pixels = np.array(img, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    pixels = deskew(pixels)
    # print(np.shape(pixels))
    pixels = np.pad(pixels, ((4, 4), (4, 4)), mode='constant')
    img_hog = hog(pixels, block_norm='L2-Hys')
    print(np.shape(img_hog))

    new_test.append(img_hog)

print(new_train[0])
print(np.shape(new_train[0]))
# for i in range(4,20):
pca = FactorAnalysis(n_components=20)
fitted_pca = pca.fit(new_train)
# r = fitted_pca.explained_variance_
# plt.plot(r)
# plt.show()
# print(fitted_pca.explained_variance_ratio_)

# new_train = fitted_pca.transform(new_train)
print(new_train[0])
print(np.shape(new_train[0]))
clf = svm.SVC(decision_function_shape='ova')
clf.fit(new_train, new_tr_lbs)
# new_test = fitted_pca.transform(new_test)
print(clf.score(new_test, new_test_label))


def purity_score(clusters, classes):

    A = np.c_[(clusters, classes)]
    n_accurate = 0.
    for j in np.unique(A[:, 0]):
        z = A[A[:, 0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]