import numpy as np
import pandas as pd
import numbers
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def normalize_data(data):
    '''
    :param data: input an array
    :return: z-score
    '''
    col_mean = np.mean(data, axis=0)
    col_std = np.std(data, axis=0)
    res = (data - col_mean) / col_std

    return res

class softmax_regression(object):

    def __init__(self, penalty='l2', C=1.0, tol=1e-4, fit_intercept=True, intercept_scaling=1, optimal_method='SGD',
                 learning_rate=0.01, batch_num=None, random_state=None, max_iter=100, min_iter=20, verbose=False,
                 warm_start=False, n_jobs=None):
        self.C = C
        self.tol = tol
        self.penalty = penalty
        self.coef = None
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.optimal_method = optimal_method
        self.learning_rate = learning_rate
        self.batch_num = batch_num
        self.random_state = random_state
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs

    def data_check(self, X, y):
        n_classes = len(np.unique(y))
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes")

    def softmax(self, x, w):
        ori_mat = np.exp(np.dot(x, w.T))
        try:
            prob_mat = ori_mat / np.sum(ori_mat, axis=1).reshape((ori_mat.shape[0], 1))
        except:
            prob_mat = ori_mat / np.sum(ori_mat)

        if True in np.isnan(prob_mat) or True in np.isinf(prob_mat):
            raise Exception('It Products inf or nan value, please check whether '
                            'needs to normalize data!')

        return prob_mat

    def initial_paras(self, dim_0, dim_1):
        '''
        :param dim_0: nums of classes
        :param dim_1: X.shape[1]
        :return: w
        '''
        w = np.zeros((dim_0, dim_1))

        return w

    def penalty_gradient(self, w, C):
        # l1正则梯度暂时没推导出，次梯度存在求解慢的问题
        if self.penalty == 'l2':
            dw = C * w
        else:
            dw = w

        return dw

    def batch_gradient_descent(self, w, X, y, learning_rate):
        for label in np.unique(y):
            # calculate H(w, X) * X
            select_X = X[y == label, :]
            H = np.array(1 - self.softmax(select_X, w)[:, label]).reshape((select_X.shape[0], 1))
            gradient_for_label = np.sum(H * select_X, axis=0) / -X.shape[0]

            penalty_dw = self.penalty_gradient(w[label, :], self.C)
            w[label, :] = w[label, :] - learning_rate * (gradient_for_label + penalty_dw)

        return w

    def stochastic_gradient_descent(self, w, X, y, learning_rate):
        for label in np.unique(y):
            # 1.shuffle train data for single label
            select_X = X[y == label, :]
            sample_size = select_X.shape[0]
            shuffle_indices = np.random.permutation(sample_size)

            # update paras
            for idx in shuffle_indices:
                H = np.array(1 - self.softmax(select_X[idx], w)[label])
                gradient_for_label = -H * select_X[idx]

                penalty_dw = self.penalty_gradient(w[label, :], self.C)
                w[label, :] = w[label, :] - learning_rate * (gradient_for_label + penalty_dw)

        return w

    def mini_batch_gradient_descent(self, w, X, y, batch_num, learning_rate):
        for label in np.unique(y):
            # 1.shuffle train data
            select_X = X[y == label, :]
            sample_size = select_X.shape[0]
            shuffle_indices = np.random.permutation(sample_size)
            batch_size = int(np.floor((sample_size / batch_num)))

            # 2.update parameters by batches
            for i in range(batch_num):
                idx_list = [shuffle_indices[i + j * batch_num] for j in range(batch_size)]
                batch_X = select_X[idx_list]

                H = np.array(1 - self.softmax(batch_X, w)[:, label]).reshape((batch_X.shape[0], 1))
                gradient_for_label = np.sum(H * batch_X, axis=0) / -batch_X.shape[0]

                penalty_dw = self.penalty_gradient(w[label, :], self.C)
                w[label, :] = w[label, :] - learning_rate * (gradient_for_label + penalty_dw)

        return w

    def solver(self, X, y, w, iter_nums, learning_rate, optimal_method, verbose=False, *args, **kwargs):
        # update parameters
        last_w = 0
        if optimal_method == 'SGD':
            # 终止条件一般有三种: 1.达到最大迭代次数; 2.前后两次梯度变化值小于某个阈值; 3.损失函数变化小于某个阈值
            # 由于SR损失函数存在log计算，很容易导致inf或者nan值，第三种终止判断条件可省略
            for i in range(iter_nums):
                adjust_learning_rate = learning_rate / (1 + i * 0.3)
                w = self.stochastic_gradient_descent(w, X, y, adjust_learning_rate)
                gradient_change = np.sum(np.abs(w - last_w))
                if gradient_change <= self.tol and i > self.min_iter:
                    print("gradient update break, the nums of iteration is %s, "
                          "and gradient change is %s" %(i, gradient_change))
                    break
                last_w = w.copy()

        elif optimal_method == 'BGD':
            for i in range(iter_nums):
                adjust_learning_rate = learning_rate / (1 + i * 0.3)
                w = self.batch_gradient_descent(w, X, y, adjust_learning_rate)
                gradient_change = np.sum(np.abs(w - last_w))
                if gradient_change <= self.tol and i > self.min_iter:
                    print("gradient update break, the nums of iteration is %s, "
                          "and gradient change is %s" % (i, gradient_change))
                    break
                last_w = w.copy()

        elif optimal_method == 'MBGD':
            batch_num = kwargs.get('batch_num')
            for i in range(iter_nums):
                adjust_learning_rate = learning_rate / (1 + i * 0.3)
                w = self.mini_batch_gradient_descent(w, X, y, batch_num, adjust_learning_rate)
                gradient_change = np.sum(np.abs(w - last_w))
                if gradient_change <= self.tol and i > self.min_iter:
                    print("gradient update break, the nums of iteration is %s, "
                          "and gradient change is %s" % (i, gradient_change))
                    break
                # array1 = array2,array2是array1的视图，指向同一地址
                last_w = w.copy()

        return w

    def fit(self, X, y):
        # 1.check init paras
        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError("Maximum number of iteration must be positive; got (max_iter=%r)" % self.max_iter)
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be positive; got (tol=%r)" % self.tol)
        if self.optimal_method not in ['SGD', 'BGD', 'MBGD']:
            raise ValueError("Please select an optimal method in ['SGD', 'BGD', 'MBGD']")
        if self.optimal_method == 'MBGD' and self.batch_num is None:
            raise ValueError("Please input an appropriate batch num for MBGD")
        if self.penalty not in ['l1', 'l2']:
            raise ValueError("Please input an appropriate penalty term in ['l1', 'l2']")

        # 2.check input data
        self.data_check(X, y)

        # 3.whether or not fitting intercept
        n_samples, n_features = X.shape
        if self.fit_intercept:
            X = np.concatenate((X, np.ones(shape=(n_samples, 1))), axis=1)

        # 4.initialize parameters(coef)
        n_classes = len(np.unique(y))
        w = self.initial_paras(dim_0=n_classes, dim_1=X.shape[1])

        # 5.solve function
        self.coef = self.solver(X, y, w, iter_nums=self.max_iter, learning_rate=self.learning_rate,
                                optimal_method=self.optimal_method, batch_num=self.batch_num)

    def predict(self, X):
        if self.coef is None:
            raise ValueError("Please fit model firstly")
        n_samples, n_features = X.shape
        if self.fit_intercept:
            X = np.concatenate((X, np.ones(shape=(n_samples, 1))), axis=1)
        pred_prod = self.softmax(X, self.coef)
        pred_label = np.argmax(pred_prod, axis=1)

        return pred_prod, pred_label

if __name__ == '__main__':
    # 1.load data
    iris = load_iris()
    X = iris['data']
    y = iris['target']

    # 2.split data
    train_X, test_X, train_y, test_y = train_test_split(X, y, shuffle=True, test_size=0.10)

    # 2.5.normalize data
    train_X = normalize_data(train_X)
    test_X = normalize_data(test_X)

    # 3.model training and print coef
    SR = softmax_regression(optimal_method='SGD', batch_num=2, max_iter=2, learning_rate=0.05)
    SR.fit(train_X, train_y)

    # 4.test data and train data fit
    train_X_pred_prod, train_X_pred_label = SR.predict(train_X)
    test_X_pred_prod, test_X_pred_label = SR.predict(test_X)

    # 5.calculate evaluation metrics of LR
    train_set_accuracy = sum(train_y == train_X_pred_label) / train_y.shape[0]
    test_set_accuracy = sum(test_y == test_X_pred_label) / test_y.shape[0]

    print("train set accuracy is %s" %train_set_accuracy)
    print("test set accuracy is %s" % test_set_accuracy)

    # 6.print coef of LR
    print("The SR model coef is %s" % SR.coef)
