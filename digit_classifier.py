from mnist import MNIST
import numpy as np
from skimage.feature import hog

mndata = MNIST('data')
tr_imgs , tr_lbs = mndata.load_training()
test_imgs , test_lbs = mndata.load_testing()



new_train = []
for img in tr_imgs :
    pixels = np.array(img, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    img_hog = hog(pixels)
    new_train.append(img_hog)



