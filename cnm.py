import h5py
import numpy as np

import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

import random
import pickle

with h5py.File('images_training.h5', 'r') as H:
    data = np.copy(H['data'])
with h5py.File('labels_training.h5', 'r') as H:
    label = np.copy(H['label'])
with h5py.File('images_testing.h5', 'r') as H:
    data_test = np.copy(H['data'])
with h5py.File('labels_testing_2000.h5', 'r') as H:
    label_test = np.copy(H['label'])

print(data_test.shape)
dropout_p = 0.4
R_P = 0.5
batch_sz = 100
lr = 0.02
max_iter = 12000

xigema = pow(10, -7)



# print(data.shape)


class BP_network(object):

    def __init__(self):
        # 784 node in input layer
        # 1
        self.w1 = np.random.randn(784, 512) * np.sqrt(2 / 784)
        self.w2 = np.random.randn(512, 256) * np.sqrt(2 / 512)
        self.w3 = np.random.randn(256, 10) * np.sqrt(2 / 256)

        self.b1 = np.zeros([1, 512])
        self.b2 = np.zeros([1, 256])
        self.b3 = np.zeros([1, 10])

        self.r1 = 0
        self.r2 = 0
        self.r3 = 0

        self.r1_b = 0
        self.r2_b = 0
        self.r3_b = 0

        self.m1_b = 0
        self.m2_b = 0
        self.m3_b = 0

        self.v1_b = 0
        self.v2_b = 0
        self.v3_b = 0

        self.m1 = 0
        self.m2 = 0
        self.m3 = 0

        self.v1 = 0
        self.v2 = 0
        self.v3 = 0
        # self.w1 = 2 * np.random.randn(784, 256) * 0.01  # (784, 256)
        # self.w2 = 2 * np.random.randn(256, 64) * 0.01  # (256, 64)
        # self.w3 = 2 * np.random.randn(64, 10) * 0.01  # (64, 10)

        # self.b1 = 2 * np.random.randn(1, 256) * 0.01  # (batch_sz, 256)
        # self.b2 = 2 * np.random.randn(1, 64) * 0.01  # (batch_sz, 64)
        # self.b3 = 2 * np.random.randn(1, 10) * 0.01  # (batch_sz, 10)

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
        x2 = ReLu(x1w2)
        x3 = softmax(x2w3)

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
        Loss = -np.log(probability)  # + np.sum(np.abs(self.w1)) + np.sum(np.abs(self.w2)) + np.sum(np.abs(self.w3))
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
        d_na_x3 = self.x3.copy()
        d_na_x3[np.arange(batch_sz), input_label] -= 1

        dw3 = (self.x2.T.dot(d_na_x3) + (2 * R_P) * self.w3)

        db3 = np.sum(d_na_x3.copy(), axis=0)

        dx2 = d_na_x3.dot(self.w3.T)
        ReLu2 = self.x2.copy()
        ReLu2[ReLu2 < 0] = 0
        ReLu2[ReLu2 > 0] = 1
        d_na_x2 = dx2 * ReLu2
        db2 = np.sum(d_na_x2.copy(), axis=0)

        dw2 = (self.x1.T.dot(d_na_x2) + (2 * R_P) * self.w2)

        # dx1 = d_na_x2.dot(self.w2.T)
        dx1 = d_na_x2.dot(self.w2.T)

        # x1_mask = self.x1_drop.copy()
        # x1_mask[x1_mask != 0] = 1
        # print(x1_mask)
        # print(np.sum(1 for i in x1_mask if i is 0))
        # print(np.sum(1 for i in x1_mask if i is 1))
        # dx1 = dx1_drop * x1_mask

        # print(dx1)
        ReLu1 = self.x1.copy()
        ReLu1[ReLu1 < 0] = 0
        ReLu1[ReLu1 > 0] = 1
        d_na_x1 = dx1 * ReLu1
        db1 = np.sum(d_na_x1.copy(), axis=0)

        dw1 = (self.x0.T.dot(d_na_x1) + (2 * R_P) * self.w1)

        '''
        self.m1 = decay_rate * self.m1 + (1 - decay_rate) * dw1
        self.m2 = decay_rate * self.m2 + (1 - decay_rate) * dw2
        self.m3 = decay_rate * self.m3 + (1 - decay_rate) * dw3

        self.v1 = decay_rate * self.v1 + (1 - decay_rate) * dw1 * dw1
        self.v2 = decay_rate * self.v2 + (1 - decay_rate) * dw2 * dw2
        self.v3 = decay_rate * self.v3 + (1 - decay_rate) * dw3 * dw3

        self.w3 -= lr * self.m3 / (0.001 + np.sqrt(self.v3))
        self.w2 -= lr * self.m2 / (0.001 + np.sqrt(self.v2))
        self.w1 -= lr * self.m1 / (0.001 + np.sqrt(self.v1)) 
        
        self.m1_b = decay_rate * self.m1_b + (1 - decay_rate) * db1
        self.m2_b = decay_rate * self.m2_b + (1 - decay_rate) * db2
        self.m3_b = decay_rate * self.m3_b + (1 - decay_rate) * db3

        self.v1_b = decay_rate * self.v1_b + (1 - decay_rate) * db1 * db1
        self.v2_b = decay_rate * self.v2_b + (1 - decay_rate) * db2 * db2
        self.v3_b = decay_rate * self.v3_b + (1 - decay_rate) * db3 * db3

        self.b3 -= lr * self.m3_b / (0.001 + np.sqrt(self.v3_b))
        self.b2 -= lr * self.m2_b / (0.001 + np.sqrt(self.v2_b))
        self.b1 -= lr * self.m1_b / (0.001 + np.sqrt(self.v1_b))    
        '''

        dw1 /= batch_sz
        dw2 /= batch_sz
        dw3 /= batch_sz
        db1 /= batch_sz
        db2 /= batch_sz
        db3 /= batch_sz

        self.r1 += dw1 * dw1
        self.r2 += dw2 * dw2
        self.r3 += dw3 * dw3

        self.w3 -= lr * (1 / (xigema + np.sqrt(self.r3))) * dw3
        self.w2 -= lr * (1 / (xigema + np.sqrt(self.r2))) * dw2
        self.w1 -= lr * (1 / (xigema + np.sqrt(self.r1))) * dw1

        self.r1_b += db1 * db1
        self.r2_b += db2 * db2
        self.r3_b += db3 * db3

        self.b3 -= lr * (1 / (xigema + np.sqrt(self.r3_b))) * db3
        self.b2 -= lr * (1 / (xigema + np.sqrt(self.r2_b))) * db2
        self.b1 -= lr * (1 / (xigema + np.sqrt(self.r1_b))) * db1

    def have_a_try(self, input_data, input_label):
        x1 = self.ReLu(np.dot(input_data, self.w1) + self.b1)
        x2 = self.ReLu(np.dot(x1, self.w2) + self.b2)  # + self.b2
        x3 = self.softmax(np.dot(x2, self.w3) + self.b3)  # + self.b3

        predict = np.argmax(x3, axis=1)  # 预测标签值
        s = predict - input_label  # 与真实标签值做对比
        num = np.sum(1 for i in s if i == 0)  # 看看预测中几个

        accuracy = num / input_data.shape[0]  # 算准确率

        probability = x3[np.arange(input_data.shape[0]), input_label]

        Loss = -np.log(probability)
        # + np.sum(np.abs(self.w1)) + np.sum(np.abs(self.w2)) + np.sum(np.abs(self.w3))

        return accuracy, Loss

    def pca(self, matrix):
        U, S, VT = np.linalg.svd(matrix, full_matrices=False)
        S = np.diag(S)
        rank = S.shape[0]
        mzz = int(0.8 * rank)
        return (U[:, :mzz].dot(S[:mzz, :mzz])).dot(VT[:mzz, :])


# data  :  [x0, x1, x2, x3]
# label :  x0 + x1 - x2 - x3 > 0? 1 : 0


if __name__ == '__main__':

    # batch size, learning rate, max iteration

    NN = BP_network()

    # define the train set and pre-process
    data = np.array(data, dtype=float)

    # for i in np.arange(data.shape[0]):
    #     data[i] = NN.pca(data[i])

    data = data.reshape(30000, -1)
    train_data = data
    train_label = label
    train_data -= np.mean(train_data, axis=0)
    train_data /= np.std(train_data, axis=0)

    # define the validation set and pre-process
    data_test = np.array(data_test, dtype=float)
    data_test = data_test.reshape(5000, 784)
    data_test -= np.mean(data_test, axis=0)
    data_test /= np.std(data_test, axis=0)
    data_labeled = data_test[0:2000]

    # 定义一会plot train loss, train acc, test loss, test acc 的横纵坐标
    x = np.arange((batch_sz * max_iter) / train_data.shape[0])
    train_loss = np.array([])
    train_acc = np.array([])
    test_acc = np.array([])
    test_loss = np.array([])

    mu = 0
    sigma = 0.01
    last_test_acc = 0
    for iters in range(max_iter):
        # shuffle it!
        if iters % (train_data.shape[0] / batch_sz) == 0:
            '''
            every (train_data.shape[0] / batch_sz) iterations, 
            the whole train set has been processed once, which was called an "epoch"
            '''
            inds = np.random.permutation(train_data.shape[0])
            train_data = train_data[inds]
            train_label = train_label[inds]

            # if (1 + iters * batch_sz / train_data.shape[0]) in (10, 20, 30):
            #     lr /= 10
            # if (1 + iters * batch_sz / train_data.shape[0]) == 15:
            #     lr /= 5
            print("epoch: " + str(1 + iters * batch_sz / train_data.shape[0]) + " the lr: " + str(lr))

        # NOW select the batch
        st_idx = int((iters % (train_data.shape[0] / batch_sz)) * batch_sz)
        ed_idx = st_idx + batch_sz
        input_data = train_data[st_idx: ed_idx].copy()
        input_label = train_label[st_idx: ed_idx].copy()

        # add gaussian noise
        for i in np.arange(input_data.shape[0]):
            input_data[i] += random.gauss(mu, sigma)

        # forward propagation
        outcome, train_accuracy, output_error = NN.forward_prop(input_data, input_label)
        # train_acc = np.append(train_acc, train_accuracy)
        # train_loss = np.append(train_loss, (np.mean(np.abs(output_error))))

        if iters % (train_data.shape[0] / batch_sz) == 0:
            # for every epoch, we output the train acc and loss
            train_acc = np.append(train_acc, train_accuracy)
            train_loss = np.append(train_loss, (np.mean(np.abs(output_error))))
            print("train accuracy: " + str(train_accuracy) + "  train loss: " + str(np.mean(np.abs(output_error))))

        NN.bp(input_label)

        if iters % (train_data.shape[0] / batch_sz) == 0:
            # for every epoch, we output the test acc and loss
            acc, Loss = NN.have_a_try(data_labeled, label_test)
            test_acc = np.append(test_acc, acc)
            test_loss = np.append(test_loss, (np.mean(np.abs(Loss))))

            # if acc >= 0.89:
            #     if acc - last_test_acc >= 0.002 and last_test_acc < 0.895:
            #         lr /= 2
            #         print("haha")
            # last_test_acc = acc

            print("test accuracy : " + str(acc) + "  test Loss: " + str(np.mean(np.abs(Loss))))

            # vw_iter = new_vw_iter
            # vb_iter = new_vb_iter

    plt.ylim(0, 2)
    plt.subplot(211)
    plt.plot(x, train_acc, x, train_loss)
    plt.subplot(212)
    plt.plot(x, test_acc, x, test_loss)
    plt.show()
