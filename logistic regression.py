import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

class logistic_regression(object):
    def sigmoid(self, x):
        res =  1 / (1 + np.exp(-x))

        return res

    def initial_paras(self, dim):
        w = np.zeros((dim, 1))
        b = 0

        return w, b

    def propagate(self, w, b, X, y):
        sample_size = X.shape[0]
        A = self.sigmoid(np.dot(X, w) + b)
        Z = A - y
        cost = np.dot(y.T, np.log(A)) + np.dot((1-y).T, np.log(1-A))

        dw = np.dot(X.T, Z) / sample_size
        db = np.sum(Z) / sample_size

        return cost, dw, db

    def optimize(self, iter_nums, learning_rate, verbose=False):
        # 1.load data
        diabetes = load_diabetes()
        X = diabetes['data']
        y = np.random.randint(0, 2, size=X.shape[0])
        y = y.reshape((y.shape[0], 1))

        # 2.initialize parameters
        w, b = self.initial_paras(dim=X.shape[1])

        # 3.update parameters
        for i in range(iter_nums):
            cost, dw, db = self.propagate(w, b, X, y)
            w = w - learning_rate * dw
            b = b - learning_rate * db
            if verbose:
                print('module cost is : ', cost)

        return w, b

    def predict(self, X, w, b):
        pass

    def __call__(self, *args, **kwargs):
        self.optimize(iter_nums=100, learning_rate=0.1, verbose=True)

class softmax_regression(object):
    '''
    softmax regression classifier.

    Parameters
    ----------
    eta: float(default: 0.01)
        Learning rate(between 0.0 and 1.0)
    epoches: int(default: 50)
        Passes over the training data
        Prior to each epoch, the dataset is shuffled.
        if 'minibatches > 1' to prevent cycles in stochastic gradient descent.

    '''
    def __init__(self, eta=0.01, epochs=50, l2=0.0, minibatches=1, n_classes=None, random_seed=None):
        self.eta = eta
        self.epochs = epochs
        self.l2 = l2
        self.minibatches = minibatches
        self.n_classes = n_classes
        self.random_seed = random_seed

    def _softmax(self, z):
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def _cross_entropy(self, output, y_target):
        return -np.sum(np.log(output) * y_target, axis=1)

    def _cost(self, cross_entropy):
        L2_term = self.l2 * np.sum(self.w_ ** 2)
        cross_entropy = cross_entropy + L2_term
        return 0.5 * np.mean(cross_entropy)

    def _one_hot(self, y, n_labels, dtype):
        '''
        Return a matrix where each sample in y is represented as a row,
        and each column represents the class label in the one-hot encoding scheme.

        Example:
            y = np.array([0, 1, 2, 3, 4, 2])
            mc = _BaseMultiClass()
            mc._one_hot(y=y, n_labels=5, dtype='float')

            np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 1, 0, 0],)

        '''
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1

        return mat.astype(dtype)

    def _yield_minibatches_idx(self, n_batches, data_array, shuffle=True):
        indices = np.array(data_array.shape[0])

        if shuffle:
            indices = np.random.permutation(indices)


    def _init_params(self, weight_shape, bias_shape=(1,), dtype='float64',
                     scale=0.01, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        w = np.random.normal(loc=0.0, scale=scale, size=weight_shape)
        b = np.zeros(shape=bias_shape)

        return b.astype(dtype), w.astype(dtype)

    def _fit(self, X, y, init_params=True):
        if init_params:
            if self.n_classes is None:
                self.n_classes = np.max(y) + 1
            self._n_features = X.shape[1]

            self.b_, self.w_ = self._init_params(weight_shape=(self._n_features, self.n_classes),
                                                 bias_shape=(self.n_classes,),
                                                 random_seed=self.random_seed)
            self.cost_ = []

        y_enc = self._one_hot(y=y, n_labels=self.n_classes, dtype=np.float)

        for i in range(self.epochs):
            for idx in self._yield_minibatches_idx()


    def fit(self, X, y, init_params=True):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self._fit(X=X, y=y, init_params=init_params)


if __name__ == '__main__':
    module = logistic_regression()
    module()
