import numpy as np
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import time

(X_train, Y_train), (x_val, y_val) = datasets.mnist.load_data()
data_size = len(X_train)
# 处理训练集和测试集，将它们铺平
X_train = np.reshape(X_train, [data_size, 28 * 28])
x_val = np.reshape(x_val, [len(x_val), 28 * 28])


def distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2, axis=1)


def predict_one(test_one, n_neighbors):
    test_mat = np.tile(test_one, (len(X_train), 1))
    a = distance(X_train, test_mat)
    c = sorted(list(zip(a, Y_train)))
    neighbors = c[0:n_neighbors]  # get k nearest neighbors
    labels = {}
    # count times of each label that occurs in the k nearest neighbors
    for neighbor in neighbors:
        label = neighbor[1]
        labels[label] = labels.get(label, 0) + 1
    return max(labels)  # return the most possible label


corr = 0.0
for i in range(len(x_val)):
    predict = predict_one(x_val[i], 10)
    if predict == y_val[i]:
        print(" step ", i, "Corr", "         ", "正确", corr, "个", "错误", i + 1 - corr)
        corr += 1
    else:
        print(" step ", i, "False", "         ", "正确", corr, "个", "错误", i + 1 - corr)
print("acc is ", corr / float(len(x_val)))
