from mnist import MNIST
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn import svm


mndata = MNIST('data')
tr_imgs , tr_lbs = mndata.load_training()
test_imgs , test_lbs = mndata.load_testing()

new_tr_lbs = [item for item in tr_lbs]
new_test_label = [item for item in test_lbs]


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

print(new_train[0])
pca = PCA(n_components=5)
pca.fit(new_train).transform(new_train)


clf = svm.SVC(decision_function_shape='ova')
clf.fit(tr_imgs, new_tr_lbs)
print(clf.score(test_imgs,new_test_label))
