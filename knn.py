from mnist import MNIST
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA ,FactorAnalysis
from sklearn.neighbors import KNeighborsClassifier
import time

start_time = time.time()

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

pca = PCA(n_components=20)
new_train = pca.fit_transform(new_train)
new_test = pca.fit_transform(new_test)


neigh = KNeighborsClassifier(n_neighbors=100)
neigh.fit(new_train, new_tr_lbs)

print('hog  knn:100')
print(neigh.score(new_test,new_test_label))


print("--- %s seconds ---" % (time.time() - start_time))
