import struct
import time
import numpy as np

path = "C://Users//Administrator//Desktop//deep_learning//data//"


def load_images(file_name):
    with open(path + file_name, 'rb') as bin_file:
        buffers = bin_file.read()
        magic, num, rows, cols = struct.unpack_from('>IIII', buffers, 0)
        bits = num * rows * cols                            # 整个images数据大小
        images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    images = np.reshape(images, [num, rows * cols])         # 变形为相应的矩阵
    return images


def load_labels(file_name):
    with open(path + file_name, 'rb') as bin_file:
        buffers = bin_file.read()
        magic, num = struct.unpack_from('>II', buffers, 0)  # 读取数据数量信息
        labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    labels = np.reshape(labels, [num])                      # 变形为一维向量
    return labels


# 读取并处理数据集
x_train, y_train = load_images("train-images.idx3-ubyte"), load_labels("train-labels.idx1-ubyte")
x_test, y_test = load_images("t10k-images.idx3-ubyte"), load_labels("t10k-labels.idx1-ubyte")
test_data = list(zip(x_test, y_test))


def fit(k):
    """
    对数据进行拟合，使用KNN算法:
    计算出临近的 K 个样本的类别，将当前样本的类别归于数量最多的样本
    """
    error = 0
    start = time.time()
    for step in range(len(x_test)):
        test, label = test_data[step][0], test_data[step][1]
        differ = np.tile(test, (len(x_train), 1)) - x_train  # 通过broadcast使得样本形成可以和训练集相减的矩阵
        distances = (differ ** 2).sum(axis=1) ** 0.5         # 计算欧氏距离，也就是L2范数
        sorted_distances = distances.argsort()               # 对距离进行排序，但是使用argsort()函数得到按照值排序的索引
        class_count = {}
        for i in range(k):
            vote_label = y_train[sorted_distances[i]]        # 取出前k个元素的类别
            class_count[vote_label] = class_count.get(vote_label, 0) + 1
        # 按照键值对字典进行排序
        sorted_class_count = sorted(class_count.items(), key=lambda d: d[1], reverse=True)
        predict = sorted_class_count[0][0]
        error += int(predict != label)
        print("已处理：{steps} 错误数：{errors} 错误率:{per} 耗时:{cost}".format(steps=step + 1, errors=error,
                                                                    per=error / (step + 1), cost=time.time() - start))
    print("KNN最终错误率为： ", error / len(x_test) * 100, "%")


if __name__ == "__main__":
    print("start!")
    fit(k=10)
