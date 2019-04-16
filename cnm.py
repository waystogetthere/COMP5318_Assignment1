import h5py
import numpy as np

import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

with h5py.File('images_training.h5', 'r') as H:
    data = np.copy(H['data'])
with h5py.File('labels_training.h5', 'r') as H:
    label = np.copy(H['label'])
with h5py.File('images_testing.h5', 'r') as H:
    data_test = np.copy(H['data'])
with h5py.File('labels_testing_2000.h5', 'r') as H:
    label_test = np.copy(H['label'])

data = data.reshape(30000, -1)

print(data_test.shape)


# print(data.shape)


class BP_network(object):
    # The network is like:
    #    l0 -> w1 -> l1 -> w2 -> (l2 == y?)
    def __init__(self):
        # 784 node in input layer
        # 1
        self.w1 = 2 * np.random.randn(784, 256) * 0.01  # (784, 256)
        self.w2 = 2 * np.random.randn(256, 64) * 0.01  # (256, 64)
        self.w3 = 2 * np.random.randn(64, 10) * 0.01  # (64, 10)

        self.b1 = 2 * np.random.randn(batch_sz, 256) * 0.01  # (batch_sz, 256)
        self.b2 = 2 * np.random.randn(batch_sz, 64) * 0.01  # (batch_sz, 64)
        self.b3 = 2 * np.random.randn(batch_sz, 10) * 0.01  # (batch_sz, 10)

    def ReLu(self, x):
        x[x < 0] = 0
        return x

    def softmax(self, x):
        z = x.copy()
        result = np.zeros(z.shape)
        for i in np.arange(x.shape[0]):
            max_mzz = np.max(x[i])
            z[i] -= max_mzz
            result[i] = np.exp(z[i]) / np.sum(np.exp(z[i]))

        return result

    def forward_prop(self, input_data, input_label):
        '''

        the net is:
        x1 = ReLu(x0w1)
        x2 = softmax(x1w2)

        '''
        self.x0 = input_data.reshape(batch_sz, 784)  # (batch_sz , 784)
        self.x1 = self.ReLu(np.array(np.dot(self.x0, self.w1) + self.b1, dtype=np.float32))  # (batch_sz, 256)
        self.x2 = self.ReLu(np.array(np.dot(self.x1, self.w2) + self.b2, dtype=np.float32))
        self.x3 = self.softmax(np.array(np.dot(self.x2, self.w3) + self.b3, dtype=np.float32))  # (batch_sz , 10)

        predict = np.argmax(self.x3, axis=1)  # 预测标签值
        s = predict - input_label  # 与真实标签值做对比
        num = np.sum(1 for i in s if i == 0)  # 看看预测中几个
        accuracy = num / input_data.shape[0]  # 算准确率

        probability = self.x3[np.arange(batch_sz), input_label]  # (batch_sz, 1)
        Loss = -np.log(probability)
        return self.x3, accuracy, Loss

    def bp_with_momentum(self, v_w, v_b, input_label):
        d_na_x3 = self.x3  # (batch_sz, 10)
        d_na_x3[np.arange(batch_sz), input_label] -= 1

        dw3 = self.x2.T.dot(d_na_x3)
        db3 = d_na_x3.copy()

        dx2 = d_na_x3.dot(self.w3.T)
        ReLu2 = self.x2.copy()
        ReLu2[ReLu2 < 0] = 0
        ReLu2[ReLu2 > 0] = 1
        d_na_x2 = dx2 * ReLu2
        db2 = d_na_x2.copy()

        dw2 = self.x1.T.dot(d_na_x2)

        dx1 = d_na_x2.dot(self.w2.T)
        ReLu1 = self.x1.copy()
        ReLu1[ReLu1 < 0] = 0
        ReLu1[ReLu1 > 0] = 1
        d_na_x1 = dx1 * ReLu1
        db1 = d_na_x1.copy()

        dw1 = self.x0.T.dot(d_na_x1)

        new_v_w = [0.9 * v_w[0] + lr * dw1, 0.9 * v_w[1] + lr * dw2, 0.9 * v_w[2] + lr * dw3]
        new_v_b = [0.9 * v_b[0] + lr * db1, 0.9 * v_b[1] + lr * db2, 0.9 * v_b[2] + lr * db3]

        self.w3 -= new_v_w[2]
        self.w2 -= new_v_w[1]
        self.w1 -= new_v_w[0]

        self.b3 -= new_v_b[2]
        self.b2 -= new_v_b[1]
        self.b1 -= new_v_b[0]

        return new_v_w, new_v_b

    def bp(self, input_label):
        d_na_x3 = self.x3  # (batch_sz, 10)
        d_na_x3[np.arange(batch_sz), input_label] -= 1

        dw3 = self.x2.T.dot(d_na_x3)
        db3 = d_na_x3.copy()

        dx2 = d_na_x3.dot(self.w3.T)
        ReLu2 = self.x2.copy()
        ReLu2[ReLu2 < 0] = 0
        ReLu2[ReLu2 > 0] = 1
        d_na_x2 = dx2 * ReLu2
        db2 = d_na_x2.copy()

        dw2 = self.x1.T.dot(d_na_x2)

        dx1 = d_na_x2.dot(self.w2.T)
        ReLu1 = self.x1.copy()
        ReLu1[ReLu1 < 0] = 0
        ReLu1[ReLu1 > 0] = 1
        d_na_x1 = dx1 * ReLu1
        db1 = d_na_x1.copy()

        dw1 = self.x0.T.dot(d_na_x1)

        self.w3 -= lr * dw3
        self.w2 -= lr * dw2
        self.w1 -= lr * dw1

        self.b3 -= lr * db3
        self.b2 -= lr * db2
        self.b1 -= lr * db1

    def have_a_try(self, input_data, input_label):
        x1 = self.ReLu(np.dot(input_data, self.w1))
        x2 = self.ReLu(np.dot(x1, self.w2))
        x3 = self.softmax(np.dot(x2, self.w3))

        predict = np.argmax(x3, axis=1)  # 预测标签值
        s = predict - input_label  # 与真实标签值做对比
        num = np.sum(1 for i in s if i == 0)  # 看看预测中几个

        accuracy = num / input_data.shape[0]  # 算准确率

        probability = x3[np.arange(input_data.shape[0]), input_label]

        Loss = -np.log(probability)

        return accuracy, Loss


# data  :  [x0, x1, x2, x3]
# label :  x0 + x1 - x2 - x3 > 0? 1 : 0


if __name__ == '__main__':

    # batch size, learning rate, max iteration
    batch_sz = 25
    lr = 0.01
    max_iter = 10000
    NN = BP_network()

    # 瞎jb做的数据预处理
    data = np.array(data, dtype=float)

    # 定义测试数据集
    train_data = data[:25000]
    train_label = label[:25000]

    # mean = np.mean(train_data, axis=0)
    # std = np.std(train_data, axis=0)

    # print(np.min(std))
    # print(mean.shape)
    train_data -= 128
    train_data /= 255

    # 对 validation set做预处理
    data_test = np.array(data_test, dtype=float)
    data_test = data_test.reshape(5000, 784)
    data_test -= 128
    data_test /= 255
    data_labeled = data_test[0:2000]
    # data_labeled = data[25000:]
    # label_test = label[25000:]

    # 定义一会plot loss 的横纵坐标,由于每训练一百次才记录一个loss，所以x也是max_iter/100
    x = np.arange(max_iter / 100)
    train_loss = np.array([])
    train_acc = np.array([])

    # 定义momentum初始值
    vw_iter = np.zeros(3)
    print(vw_iter)
    vb_iter = np.zeros(3)
    #

    # list_x = np.linspace(0, max_iter, num=max_iter / 100)
    # print(list_x.shape)
    test_acc = np.array([])
    test_loss = np.array([])

    for iters in range(max_iter):
        # shuffle it!
        if iters % 1000 == 0: # every 1000 iterations, the whole train set has been processed once
            inds = np.random.permutation(train_data.shape[0])
            train_data = train_data[inds]
            train_label = train_label[inds]

        # NOW select the batch
        st_idx = int((iters % (train_data.shape[0] / batch_sz)) * batch_sz)
        ed_idx = st_idx + batch_sz
        input_data = train_data[st_idx: ed_idx]
        input_label = train_label[st_idx: ed_idx]

        # forward propagation
        outcome, train_accuracy, output_error = NN.forward_prop(input_data, input_label)
        if iters % 100 == 0:
            train_acc = np.append(train_acc, train_accuracy)
            train_loss = np.append(train_loss, (np.mean(np.abs(output_error))))
            print("the train accuracy is: " + str(train_accuracy))
            print("The train loss is %f" % (np.mean(np.abs(output_error))))

        # new_vw_iter, new_vb_iter = NN.bp_with_momentum(vw_iter, vb_iter, input_label)
        NN.bp(input_label)

        if iters % 100 == 0:
            acc, Loss = NN.have_a_try(data_labeled, label_test)
            test_acc = np.append(test_acc, acc)
            test_loss = np.append(test_loss, (np.mean(np.abs(Loss))))
            print("the test accuracy is: " + str(acc))
            print("The test Loss is %f" % (np.mean(np.abs(Loss))))

        # vw_iter = new_vw_iter
        # vb_iter = new_vb_iter

    # plt.plot(x, y)
    # print(list_acc)
    # print(list_loss)

    # sns.set_style('darkgrid')

    plt.subplot(211)
    plt.plot(x, train_acc, x, train_loss)
    plt.subplot(212)
    plt.plot(x, test_acc, x, test_loss)
    plt.show()
