import numpy as np
import pandas as pd
import numbers
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

class logistic_regression(object):

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
        if n_classes != 2:
            raise ValueError("This solver needs samples of 2 classes")

    def sigmoid(self, x):
        res =  1 / (1 + np.exp(-x))


        # SGD时，res可能为单元素 numpy.bool_，无法使用in判断（not iterable）
        if True in np.array(np.isnan(res)) or True in np.array(np.isinf(res)):
            raise Exception('It Products inf or nan value, please check whether '
                            'needs to normalize data!')

        return res

    def initial_paras(self, dim):
        w = np.zeros((dim,))

        return w

    def penalty_gradient(self, w, C):
        w_size = w.shape[0]
        # l1正则梯度暂时没推导出，次梯度存在求解慢的问题
        if self.penalty == 'l2':
            dw =  C * w / w_size
        else:
            dw = w

        return dw

    def batch_gradient_descent(self, w, X, y, learning_rate):
        sample_size = X.shape[0]
        A = self.sigmoid(np.dot(X, w))
        Z = A - y

        dw = np.sum(np.array([X[i] * Z[i] for i in range(sample_size)]), axis=0) / sample_size
        penalty_dw = self.penalty_gradient(w, self.C)
        w = w - learning_rate * (dw + penalty_dw)

        return w

    def stochastic_gradient_descent(self, w, X, y, learning_rate):
        # 1.shuffle train data
        sample_size = X.shape[0]
        shuffle_indices = np.random.permutation(list(range(sample_size)))

        # 2.update w and b
        for idx in shuffle_indices:
            A = self.sigmoid(np.dot(X[idx, :], w))
            Z = A - y[idx]

            dw = X[idx] * Z
            penalty_dw = self.penalty_gradient(w, self.C)
            w = w - learning_rate * (dw + penalty_dw)

        return w

    def mini_batch_gradient_descent(self, w, X, y, batch_num, learning_rate):
        # 1.shuffle train data
        sample_size = X.shape[0]
        shuffle_indices = np.random.permutation(list(range(sample_size)))
        batch_size = int(np.floor((sample_size / batch_num)))

        # 2.update parameters by batches
        for i in range(batch_num):
            idx_list = [shuffle_indices[i + j * batch_num] for j in range(batch_size)]
            batch_X = X[idx_list]
            batch_y = y[idx_list]

            A = self.sigmoid(np.dot(batch_X, w))
            Z = A - batch_y

            dw = np.sum(np.array([batch_X[i] * Z[i] for i in range(batch_size)]), axis=0) / batch_size
            penalty_dw = self.penalty_gradient(w, self.C)
            w = w - learning_rate * (dw + penalty_dw)

        return w

    def solver(self, X, y, w, iter_nums, learning_rate, optimal_method, verbose=False, *args, **kwargs):
        # update parameters
        last_w = 0
        if optimal_method == 'SGD':
            # 终止条件一般有三种: 1.达到最大迭代次数; 2.前后两次梯度变化值小于某个阈值; 3.损失函数变化小于某个阈值
            # 由于LR损失函数存在log计算，很容易导致inf或者nan值，第三种终止判断条件可省略
            for i in range(iter_nums):
                learning_rate = learning_rate / (1 + i * 0.3)
                w = self.stochastic_gradient_descent(w, X, y, learning_rate)
                gradient_change = np.sum(np.abs(w - last_w))
                if gradient_change <= self.tol and i > self.min_iter:
                    print("gradient update break, the nums of iteration is %s, "
                          "and gradient change is %s" %(i, gradient_change))
                    break
                last_w = w.copy()

        elif optimal_method == 'BGD':
            for i in range(iter_nums):
                learning_rate = learning_rate / (1 + i * 0.3)
                w = self.batch_gradient_descent(w, X, y, learning_rate)
                gradient_change = np.sum(np.abs(w - last_w))
                if gradient_change <= self.tol and i > self.min_iter:
                    print("gradient update break, the nums of iteration is %s, "
                          "and gradient change is %s" % (i, gradient_change))
                    break
                last_w = w.copy()

        elif optimal_method == 'MBGD':
            batch_num = kwargs.get('batch_num')
            for i in range(iter_nums):
                learning_rate = learning_rate / (1 + i * 0.3)
                w = self.mini_batch_gradient_descent(w, X, y, batch_num, learning_rate)
                gradient_change = np.sum(np.abs(w - last_w))
                if gradient_change <= self.tol and i > self.min_iter:
                    print("gradient update break, the nums of iteration is %s, "
                          "and gradient change is %s" % (i, gradient_change))
                    break
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
        w = self.initial_paras(dim=X.shape[1])

        # 5.solve function
        self.coef = self.solver(X, y, w, iter_nums=self.max_iter, learning_rate=self.learning_rate,
                                optimal_method=self.optimal_method, batch_num=self.batch_num)

    def predict(self, X):
        if self.coef is None:
            raise ValueError("Please fit model firstly")
        n_samples, n_features = X.shape
        if self.fit_intercept:
            X = np.concatenate((X, np.ones(shape=(n_samples, 1))), axis=1)
        pred_prod = self.sigmoid(np.dot(X, self.coef))
        pred_label = np.array(pred_prod >= 0.5, dtype='int')

        return pred_prod, pred_label

if __name__ == '__main__':
    '''
    1.问题
       模型效果很大程度上取决于learning rate，即alpha。起初未设置动态调整alpha，只有SGD的模型效果比较好;引入动态
    调整alhpa策略后，MBDG效果改善，BGD效果仍然很差。
    
    2.debug分析
       经过debug分析后，造成上述情况的原因为alpha:若不采用动态调整alpha策略，由于BGD每次更新参数采用全部数据，步幅
    较大会导致后期参数不收敛，而MBGD由于采用batch策略，稍微缓解了后期这种全部数据更新参数的弊端，SGD同理。
       引入动态更新策略后，初期效果仍然不好的原因为:随着迭代次数增加，alpha衰减过快，导致模型未能很好地进行迭代学习。
    
    3.解决策略
       最终解决策略为:合理设置alpha衰减策略，既要解决收敛问题，又不能过快衰减。
    '''
    # 1.load data
    breast_cancer = load_breast_cancer()
    X = breast_cancer['data']
    y = breast_cancer['target']

    # 2.split data
    train_X, test_X, train_y, test_y = train_test_split(X, y, shuffle=True, test_size=0.10)

    # 2.5.normalize data
    train_X = normalize_data(train_X)
    test_X = normalize_data(test_X)

    # 3.model training and print coef
    LR = logistic_regression(optimal_method='SGD', batch_num=2, max_iter=100)
    LR.fit(train_X, train_y)

    # 4.test data and train data fit
    train_X_pred_prod, train_X_pred_label = LR.predict(train_X)
    test_X_pred_prod, test_X_pred_label = LR.predict(test_X)

    # 5.calculate evaluation metrics of LR
    train_set_accuracy = sum(train_y == train_X_pred_label) / train_y.shape[0]
    train_set_precision = np.sum(
        (np.array(train_y == 1, dtype='int') + np.array(train_X_pred_label == 1, dtype='int')) == 2) / np.sum(train_X_pred_label == 1)
    train_set_recall = np.sum(
        (np.array(train_y == 1, dtype='int') + np.array(train_X_pred_label == 1, dtype='int')) == 2) / np.sum(train_y == 1)
    train_set_f1_score = 2 / (1 / train_set_precision + 1 / train_set_recall)

    test_set_accuracy = sum(test_y == test_X_pred_label) / test_y.shape[0]
    test_set_precision = np.sum(
        (np.array(test_y == 1, dtype='int') + np.array(test_X_pred_label == 1, dtype='int')) == 2) / np.sum(test_X_pred_label == 1)
    test_set_recall = np.sum(
        (np.array(test_y == 1, dtype='int') + np.array(test_X_pred_label == 1, dtype='int')) == 2) / np.sum(test_y == 1)
    test_set_f1_score = 2 / (1 / test_set_precision + 1 / test_set_recall)

    print("train set accuracy is %s, precision is %s, recall is %s, f1_score is %s"
          %(train_set_accuracy, train_set_precision, train_set_recall, train_set_f1_score))
    print("test set accuracy is %s, precision is %s, recall is %s, f1_score is %s"
          % (test_set_accuracy, test_set_precision, test_set_recall, test_set_f1_score))

    # 6.print coef of LR
    print("The LR model coef is %s" %LR.coef)