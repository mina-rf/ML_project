import cv2

import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST
from skimage.feature import hog
from sklearn import svm, discriminant_analysis
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA


def deskew(img):
    SZ = 20
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11'] / m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def feature_extraction(images):
    feature_extracted_images = []
    max_len = 30
    for img in images:
        pixels = np.array(img, dtype='uint8')
        pixels = pixels.reshape((28, 28))
        # pixels = deskew(pixels)
        # print(np.shape(pixels))
        # pixels = np.pad(pixels, ((4, 4), (4, 4)), mode='constant')
        img_features = hog(pixels, block_norm='L2-Hys')

        # sift = cv2.xfeatures2d.SIFT_create()
        # img_features = np.array([p.pt for p in sift.detect(pixels)])
        # print(np.shape(img_features))

        # feature_extracted_images.append(np.pad(img_features, (0, max_len - len(img_features)), mode='constant'))
        feature_extracted_images.append(img_features)

    return feature_extracted_images


def plot_pca(train_images, n_components):
    print("plotting")
    print(np.shape(train_images))
    pca = PCA(n_components=n_components)
    print('here')
    pca.fit(train_images)
    print('there')
    r = pca.explained_variance_ratio_
    print(r)
    plt.plot(r)
    plt.show()


def dimensionality_reduction(train_images, test_images, DR):
    pca = DR(n_components=20)
    pca.fit(train_images)
    return pca.transform(train_images), pca.transform(test_images)


def classify(train, tr_lbs, test, test_lbs):
    clf = svm.SVC(decision_function_shape='ova')
    print('svm')
    clf.fit(train, tr_lbs)
    print('svm2')
    print(clf.score(test, test_lbs))


def main():
    mndata = MNIST('data', return_type='numpy')
    tr_imgs, tr_lbs = mndata.load_training()
    test_imgs, test_lbs = mndata.load_testing()

    fe_tr_imgs = feature_extraction(tr_imgs)
    fe_test_imgs = feature_extraction(test_imgs)

    # print(np.shape(fe_tr_imgs), np.shape(fe_test_imgs))
    # plot_pca(fe_tr_imgs,81)


    # de_tr_imgs, de_test_imgs = dimensionality_reduction(fe_tr_imgs, fe_test_imgs, FactorAnalysis)
    # print(de_test_imgs[0])
    # print(np.shape(de_tr_imgs), np.shape(de_test_imgs))
    # print('done with pca', np.shape(de_test_imgs))
    # classify(de_tr_imgs, tr_lbs, de_test_imgs, test_lbs)

    clf = discriminant_analysis.LinearDiscriminantAnalysis()
    clf.fit(fe_tr_imgs, tr_lbs)
    print(clf.score(fe_test_imgs, test_lbs))

    # clf = svm.SVC(decision_function_shape='ova')
    # clf.fit(de_tr_imgs, tr_lbs)
    #
    # print(clf.score(de_test_imgs, test_lbs))


if __name__ == '__main__':
    main()
