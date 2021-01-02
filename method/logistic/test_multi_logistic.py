# ================================================================
#
#   Editor      : Pycharm
#   File name   : test_multi_logistic
#   Author      : HuangWei
#   Created date: 2020-12-13 16:43
#   Email       : 446296992@qq.com
#   Description : 
#   
#    ( ˶˙º˙˶ )୨  Have Fun!!!
# ================================================================
import random
import numpy as np
import matplotlib.pyplot as plt

x1, y1 = np.random.normal(0, 1.5, 500), np.random.normal(15, 1.5, 500)
x2, y2 = np.random.normal(-3.5, 1.5, 500), np.random.normal(-12.5, 1.5, 500)
x3, y3 = np.random.normal(12.5, 1.5, 500), np.random.normal(-8, 4.5, 500)

data_1 = np.array([[x1[i], y1[i], 1] for i in range(500)])
data_2 = np.array([[x2[i], y2[i], 1] for i in range(500)])
data_3 = np.array([[x3[i], y3[i], 1] for i in range(500)])
label_1 = np.array([1 for _ in range(500)])
label_0 = np.array([0 for _ in range(500)])


def gd(data, label):
    lr = 0.01  # 学习率
    count = 0  # 当前已经迭代次数 Current number of iter
    theta = np.ones([3, 1])  # 初始化theta， 分别代表 w1, w2, b init the theta, denote w1, w2, b
    while count < 6500:
        h_theta_x = 1 / (1 + np.exp(-np.dot(data, theta)))
        label = np.reshape(label, [1500, 1])
        cost = np.mean((label - h_theta_x) * data, axis=0).reshape([-1, 1])
        theta = theta + lr * cost  # 按照公式更新theta renew the theta according to equation
        count += 1  # 新增迭代次数
    return theta


def predict(theta: np.ndarray, dataset: np.ndarray) -> float:
    return (-(theta[0] * ((dataset - 0) / 1) + theta[2]) / theta[1]) * 1 + 0


# 绘制图片，包括散点图以及中心点
def show_pic(theta1, theta2, theta3):
    plt.scatter([x for x in x1], [x for x in y1], s=30, alpha=0.7)
    plt.scatter([x for x in x2], [x for x in y2], s=30, alpha=0.7)
    plt.scatter([x for x in x3], [x for x in y3], s=30, alpha=0.7)
    x_1 = np.array([-5, 15])
    x_2 = np.array([-5, 15])
    x_3 = np.array([-2, 4])
    y_1 = predict(theta1, x_1)
    y_2 = predict(theta2, x_2)
    y_3 = predict(theta3, x_3)
    plt.plot(x_1, y_1)
    plt.plot(x_2, y_2)
    plt.plot(x_3, y_3)
    # plt.show()
    # x = np.array([20, 55])
    # y_1 = predict(theta1, x)
    # y_2 = predict(theta2, x)
    # y_3 = predict(theta3, x)
    # plt.plot(x, y)
    plt.show()


dataset = np.vstack([data_1, data_2, data_3])
first_label = np.hstack([label_1, label_0, label_0])
second_label = np.hstack([label_0, label_1, label_0])
third_label = np.hstack([label_0, label_0, label_1])

theta1 = gd(dataset, first_label)
theta2 = gd(dataset, second_label)
theta3 = gd(dataset, third_label)
show_pic(theta1, theta2, theta3)
