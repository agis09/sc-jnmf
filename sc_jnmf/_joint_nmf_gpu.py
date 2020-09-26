import cupy as cp


class Joint_NMF_GPU(object):
    def __init__(
            self, D1, D2, W1, W2, H,
            lambda1, lambda2, lambda3, lambda4,
            iter_num=100, conv_judge=1e-5, calc_log=[], regularization='l1'):
        self.D1 = cp.asarray(D1)
        self.D2 = cp.asarray(D2)
        self.W1 = cp.asarray(W1)
        self.W2 = cp.asarray(W2)
        self.H = cp.asarray(H)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.iter_num = iter_num
        self.conv_judge = conv_judge
        self.calc_log = calc_log
        self.regularization = 'l1'

    def __update_W1(self):
        if self.regularization == 'l1':
            self.W1 *= cp.divide(self.D1.dot(self.H.T) - self.lambda2 / 2,
                                 self.H.dot(self.W1.dot(self.H).T).T)
        elif self.regularization == 'l2':
            self.W1 *= cp.divide(self.D1.dot(self.H.T),
                                 self.H.dot(self.W1.dot(self.H).T).T + self.lambda2 * self.W1)

    def __update_W2(self):
        if self.regularization == 'l1':
            self.W2 *= cp.divide(self.lambda1 * self.D2.dot(self.H.T) - self.lambda3 / 2,
                                 self.lambda1 * self.H.dot(self.W2.dot(self.H).T).T)
        elif self.regularization == 'l2':
            self.W2 *= cp.divide(self.lambda1 * self.D2.dot(self.H.T),
                                 self.lambda1 * self.H.dot(self.W2.dot(self.H).T).T + self.lambda3 * self.W2)

    def __update_H(self):
        if self.regularization == 'l1':
            self.H *= cp.divide(self.D1.T.dot(self.W1).T + self.lambda1 * self.D2.T.dot(self.W2).T - self.lambda4 / 2,
                                self.W1.T.dot(self.W1).dot(self.H) + self.lambda1 * self.W2.T.dot(self.W2).dot(self.H))
        elif self.regularization == 'l2':
            self.H *= cp.divide(self.D1.T.dot(self.W1).T + self.lambda1 * self.D2.T.dot(self.W2).T,
                                self.W1.T.dot(self.W1).dot(self.H) + self.lambda1 * self.W2.T.dot(self.W2).dot(self.H) + self.lambda4 * self.H)

    def __calc_min_func(self):

        if self.regularization == 'l1':
            return cp.linalg.norm(self.D1 - self.W1.dot(self.H), ord='fro')**2 \
                + self.lambda1 * \
                cp.linalg.norm(self.D2 - self.W2.dot(self.H), ord='fro')**2 \
                + self.lambda2 * cp.linalg.norm(self.W1, ord=1) \
                + self.lambda3 * cp.linalg.norm(self.W2, ord=1) \
                + self.lambda4 * cp.linalg.norm(self.H, ord=1)
        elif self.regularization == 'l2':
            return cp.linalg.norm(self.D1 - self.W1.dot(self.H), ord='fro')**2 \
                + self.lambda1 * \
                cp.linalg.norm(self.D2 - self.W2.dot(self.H), ord='fro')**2 \
                + self.lambda2 * cp.linalg.norm(self.W1, ord='fro')**2 \
                + self.lambda3 * cp.linalg.norm(self.W2, ord='fro')**2 \
                + self.lambda4 * cp.linalg.norm(self.H, ord='fro')**2

    def calc(self):
        pre_min_func = self.__calc_min_func()

        for cnt in range(self.iter_num):
            self.__update_W1()
            self.W1[self.W1 < cp.finfo(self.W1.dtype).eps] = cp.finfo(
                self.W1.dtype).eps

            self.__update_W2()
            self.W2[self.W2 < cp.finfo(self.W2.dtype).eps] = cp.finfo(
                self.W2.dtype).eps

            self.__update_H()
            self.H[self.H < cp.finfo(self.H.dtype).eps] = cp.finfo(
                self.H.dtype).eps

            min_func = self.__calc_min_func()
            self.calc_log.append(min_func)

            if cnt > 10 and pre_min_func - min_func < self.conv_judge:
                break
            pre_min_func = min_func
        return cp.asnumpy(self.W1), cp.asnumpy(self.W2), cp.asnumpy(self.H)
