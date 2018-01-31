from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin, euclidean_distances


class Kmeans:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        n_iterations = 300
        min_cost = np.Inf
        for j in range(10):
            means = [X[i] for i in np.random.choice(len(X), self.n_components, replace=False)]
            print(means)
            for i in range(n_iterations):
                nearest = self.find_nearest(X, means)
                print(nearest)
                means = self.find_means(nearest)
                print(means)
            print([euclidean_distances(means[i], v) for i, v in enumerate(nearest)])
            cost = np.sum([np.sum(euclidean_distances(means[i], v)) for i, v in enumerate(nearest)])
            if cost < min_cost:
                self.clusters = nearest
                self.means = means

    def find_nearest(self, X, means):
        nearest = []
        for _ in means:
            nearest.append([])
        print(means)
        index = pairwise_distances_argmin(X, means)
        print(index)
        for i, ind in enumerate(index):
            nearest[ind].append(X[i])
        return nearest

    def predict(self, Y):
        index = pairwise_distances_argmin(Y, self.means)
        return index

    def find_means(self, nearest):
        means = []
        for v in nearest:
            means.append(np.mean(v, axis=0))

        return means


if __name__ == '__main__':
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    kmeans = Kmeans(n_components=2)
    kmeans.fit(X)
    print(kmeans.predict(X))
