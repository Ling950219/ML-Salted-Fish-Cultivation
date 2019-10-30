import numpy as np
import pandas as pd
import numbers
import random
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.datasets import load_breast_cancer
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

class FM(object):

    def __init__(self, penalty='l2', C=1.0, tol=1e-4, fit_intercept=True, intercept_scaling=1, optimal_method='SGD',
                 learning_rate=0.01, batch_num=None, random_state=None, max_iter=100, min_iter=20, v_len=7):
        self.C = C
        self.tol = tol
        self.penalty = penalty
        self.w = None
        self.V = None
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.optimal_method = optimal_method
        self.learning_rate = learning_rate
        self.batch_num = batch_num
        self.random_state = random_state
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.v_len = v_len

    def data_check(self, X, y):
        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This solver needs samples of at least 2 classes")

    def sigmoid(self, x):
        res =  1 / (1 + np.exp(-x))

        # SGD时，res可能为单元素 numpy.bool_，无法使用in判断（not iterable）
        if True in np.array(np.isnan(res)) or True in np.array(np.isinf(res)):
            raise Exception('It Products inf or nan value, please check whether '
                            'needs to normalize data!')

        return res

    def initial_paras(self, dim, v_len=None):
        w = np.zeros((dim,))
        V = np.ones((dim, v_len))
        for i in range(dim):
            for j in range(v_len):
                V[i, j] = random.normalvariate(0, 0.2)

        return w, V

    def cross_term(self, X, V):
        res = np.zeros(shape=(X.shape[0], ))
        for idx in range(X.shape[0]):
            x = X[idx, :]
            V_x = V * np.array(x).reshape((x.shape[0], 1))
            left = np.sum(V_x, axis=0) ** 2
            right = np.sum(V_x ** 2, axis=0)
            res[idx] = np.sum(0.5 * (left - right))

        return res

    def predict_y(self, X, w, V):
        y_pred = np.dot(X, w) + self.cross_term(X, V)


        return y_pred

    def penalty_gradient(self, w, C, sample_size):
        # l1正则梯度暂时没推导出，次梯度存在求解慢的问题
        if self.penalty == 'l2':
            dw =  C * w / sample_size
        else:
            dw = w

        return dw

    def batch_gradient_descent(self, w, X, y, V, learning_rate):
        sample_size = X.shape[0]
        y_pred = self.predict_y(X, w, V)
        y_y_pred = np.multiply(y, y_pred)
        grad_prefix = -np.multiply(1 - self.sigmoid(y_y_pred), y)
        penalty_dw = self.penalty_gradient(w, self.C, sample_size)
        dw = np.sum(X * grad_prefix.reshape((grad_prefix.shape[0], 1)), axis=0) / sample_size
        w = w - learning_rate * (dw + penalty_dw)

        n, k = V.shape
        for i in range(n):
            for f in range(k):
                dv_if = np.sum(np.dot(X, V[:, f]) * X[:, i] - V[i, f] * X[:, i] ** 2, axis=0) / sample_size
                penalty_dv_if = self.penalty_gradient(V[i, f], self.C, sample_size)
                V[i, f] = np.sum(V[i, f] - learning_rate * (dv_if + penalty_dv_if) * grad_prefix) / sample_size

        return w, V

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

                penalty_dw = self.penalty_gradient(w[label, :], self.C, 1)
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

                penalty_dw = self.penalty_gradient(w[label, :], self.C, batch_size)
                w[label, :] = w[label, :] - learning_rate * (gradient_for_label + penalty_dw)

        return w

    def solver(self, X, y, w, V, iter_nums, learning_rate, optimal_method, verbose=False, *args, **kwargs):
        # update parameters
        last_w = 0
        last_V = 0
        if optimal_method == 'SGD':
            # 终止条件一般有三种: 1.达到最大迭代次数; 2.前后两次梯度变化值小于某个阈值; 3.损失函数变化小于某个阈值
            # 由于SR损失函数存在log计算，很容易导致inf或者nan值，第三种终止判断条件可省略
            for i in range(iter_nums):
                adjust_learning_rate = learning_rate / np.sqrt(i + 1)
                w = self.stochastic_gradient_descent(w, X, y, adjust_learning_rate)
                gradient_change = np.sum(np.abs(w - last_w))
                if gradient_change <= self.tol and i > self.min_iter:
                    print("gradient update break, the nums of iteration is %s, "
                          "and gradient change is %s" %(i, gradient_change))
                    break
                last_w = w.copy()

        elif optimal_method == 'BGD':
            for i in range(iter_nums):
                adjust_learning_rate = learning_rate / np.sqrt(i + 1)
                w, V = self.batch_gradient_descent(w, X, y, V, adjust_learning_rate)
                gradient_change = np.sum(np.abs(w - last_w)) + np.sum((np.abs(V - last_V)))
                if gradient_change <= self.tol and i > self.min_iter:
                    print("gradient update break, the nums of iteration is %s, "
                          "and gradient change is %s" % (i, gradient_change))
                    break
                last_w = w.copy()
                last_V = V.copy()

        elif optimal_method == 'MBGD':
            batch_num = kwargs.get('batch_num')
            for i in range(iter_nums):
                adjust_learning_rate = learning_rate / np.sqrt(i + 1)
                w = self.mini_batch_gradient_descent(w, X, y, batch_num, adjust_learning_rate)
                gradient_change = np.sum(np.abs(w - last_w))
                if gradient_change <= self.tol and i > self.min_iter:
                    print("gradient update break, the nums of iteration is %s, "
                          "and gradient change is %s" % (i, gradient_change))
                    break
                # array1 = array2,array2是array1的视图，指向同一地址
                last_w = w.copy()

        return w, V

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
        w, V = self.initial_paras(X.shape[1], self.v_len)

        # 5.solve function
        self.w, self.V = self.solver(X, y, w, V, iter_nums=self.max_iter, learning_rate=self.learning_rate,
                                     optimal_method=self.optimal_method, batch_num=self.batch_num)

    def predict(self, X):
        if self.w is None or self.V is None:
            raise ValueError("Please fit model firstly")
        n_samples, n_features = X.shape
        if self.fit_intercept:
            X = np.concatenate((X, np.ones(shape=(n_samples, 1))), axis=1)

        y_pred = self.predict_y(X, self.w, self.V)
        y_prod = self.sigmoid(y_pred)
        y_label = np.array(y_prod >= 0.5, dtype='int')
        y_label[y_label==0] = -1

        return y_prod, y_label

if __name__ == '__main__':
    '''
        FM如果不做筛选，选择全部二次项，似乎效果可能不如LR（至少在该数据集效果不如LR）。
    '''
    # 1.load data
    breast_cancer = load_breast_cancer()
    X = breast_cancer['data']
    y = breast_cancer['target']
    y[y==0] = -1

    # 2.split data
    train_X, test_X, train_y, test_y = train_test_split(X, y, shuffle=True, test_size=0.10)

    # 2.5.normalize data
    train_X = normalize_data(train_X)
    test_X = normalize_data(test_X)

    # 3.model training and print coef
    model = FM(optimal_method='BGD', batch_num=2, max_iter=2000)
    model.fit(train_X, train_y)

    # 4.test data and train data fit
    train_X_pred_prod, train_X_pred_label = model.predict(train_X)
    test_X_pred_prod, test_X_pred_label = model.predict(test_X)

    # 5.calculate evaluation metrics of LR
    train_set_accuracy = sum(train_y == train_X_pred_label) / train_y.shape[0]
    train_set_precision = np.sum(
        (np.array(train_y == 1, dtype='int') + np.array(train_X_pred_label == 1, dtype='int')) == 2) / np.sum(
        train_X_pred_label == 1)
    train_set_recall = np.sum(
        (np.array(train_y == 1, dtype='int') + np.array(train_X_pred_label == 1, dtype='int')) == 2) / np.sum(
        train_y == 1)
    train_set_f1_score = 2 / (1 / train_set_precision + 1 / train_set_recall)

    test_set_accuracy = sum(test_y == test_X_pred_label) / test_y.shape[0]
    test_set_precision = np.sum(
        (np.array(test_y == 1, dtype='int') + np.array(test_X_pred_label == 1, dtype='int')) == 2) / np.sum(
        test_X_pred_label == 1)
    test_set_recall = np.sum(
        (np.array(test_y == 1, dtype='int') + np.array(test_X_pred_label == 1, dtype='int')) == 2) / np.sum(test_y == 1)
    test_set_f1_score = 2 / (1 / test_set_precision + 1 / test_set_recall)

    print("train set accuracy is %s, precision is %s, recall is %s, f1_score is %s"
          % (train_set_accuracy, train_set_precision, train_set_recall, train_set_f1_score))
    print("test set accuracy is %s, precision is %s, recall is %s, f1_score is %s"
          % (test_set_accuracy, test_set_precision, test_set_recall, test_set_f1_score))

    # 6.print coef of LR
    print("The fm model coef is %s %s" %(model.w, model.V))
