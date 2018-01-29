import numpy as np
from mnist import MNIST
from skimage.feature import hog
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

mndata = MNIST('data')
tr_imgs, tr_lbs = mndata.load_training()
test_imgs, test_lbs = mndata.load_testing()

new_tr_lbs = [item for item in tr_lbs]
new_test_label = [item for item in test_lbs]

new_train = []
for img in tr_imgs:
    pixels = np.array(img, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    img_hog = hog(pixels)
    new_train.append(img_hog)

new_test = []
for img in test_imgs:
    pixels = np.array(img, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    img_hog = hog(pixels)
    new_test.append(img_hog)

print(new_train[0])
print(np.shape(new_train[0]))
# for i in range(4,20):
pca = PCA(n_components=10)
fitted_pca = pca.fit(new_train)
r = fitted_pca.explained_variance_
# plt.plot(r)
# plt.show()
print(fitted_pca.explained_variance_ratio_)

new_train = fitted_pca.transform(new_train)
print(new_train[0])
print(np.shape(new_train[0]))
clf = svm.SVC(decision_function_shape='ova')
clf.fit(new_train, new_tr_lbs)
new_test = fitted_pca.transform(new_test)
print(clf.score(new_test, new_test_label))


def purity_score(clusters, classes):

    A = np.c_[(clusters, classes)]
    n_accurate = 0.
    for j in np.unique(A[:, 0]):
        z = A[A[:, 0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]