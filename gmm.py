import numpy as np


class GMM:
    def __init__(self, n_clusters):
        self.k = n_clusters

        self.means = np.zeros(n_clusters)
        self.covs = np.zeros(n_clusters)
        self.coeffs = np.zeros(n_clusters)

    def get_normal(self, s, mu, X):
        return np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1] / 2.) \
               * np.exp(-.5 * np.einsum('ij, ij -> i', X - mu, np.dot(np.linalg.inv(s), (X - mu).T).T))

    def fit(self, X):
        n_iterations = 300

        n, d = X.shape

        means = [X[i] for i in np.random.choice(len(X), self.k, replace=False)]
        covs = [np.eye(d) for _ in range(self.k)]
        coef = [1. / self.k for _ in range(self.k)]
        resp = np.zeros((n, self.k))

        for i in range(n_iterations):
            for k in range(self.k):
                # print(covs[k])
                resp[:, k] = coef[k] * self.get_normal(covs[k], means[k], X)

            resp = (resp.T / np.sum(resp, axis=1)).T

            Nk = np.sum(resp, axis=0)

            for k in range(self.k):
                means[k] = 1 / Nk[k] * np.sum(resp[:, k] * X.T, axis=1).T
                x_mean = np.matrix(X - means[k])
                covs[k] = np.array(1 / Nk[k] * np.dot(np.multiply(x_mean.T, resp[:, k]), x_mean))
                coef[k] = 1. / n * Nk[k]

        self.means = means
        self.covs = covs
        self.coeffs = coef
        self.resp = resp

    def predict(self, Y):
        labels = []
        for y in Y:
            l = np.argmax([self.get_normal(self.covs[k], self.means[k], np.expand_dims(y,axis=0)) * self.coeffs[k] for k in range(self.k)])
            labels.append(l)
        return labels


if __name__ == '__main__':
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    gmm = GMM(n_clusters=2)
    gmm.fit(X)
    # print(kmeans.predict(X))
