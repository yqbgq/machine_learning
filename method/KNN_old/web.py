from numpy import *
import time
import struct
from numpy import array
import numpy as np

BASE_PATH = "C://Users//Administrator//Desktop//deep_learning//data//"

def load_images(file_name):
    #   在读取或写入一个文件之前，你必须使用 Python 内置open()函数来打开它
    #   file object = open(file_name [, access_mode][, buffering])
    #   file_name是包含您要访问的文件名的字符串值。
    #   access_mode指定该文件已被打开，即读，写，追加等方式。
    #   0表示不使用缓冲，1表示在访问一个文件时进行缓冲。
    #   这里rb表示只能以二进制读取的方式打开一个文件
    binfile = open(BASE_PATH + file_name, 'rb')
    #   从一个打开的文件读取数据
    buffers = binfile.read()
    #   读取image文件前4个整型数字
    magic, num, rows, cols = struct.unpack_from('>IIII', buffers, 0)
    #   整个images数据大小为60000*28*28
    bits = num * rows * cols
    #   读取images数据
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    #   关闭文件
    binfile.close()
    #   转换为[60000,784]型数组
    images = np.reshape(images, [num, rows * cols])
    return images


def load_labels(file_name):
    #   打开文件
    binfile = open(BASE_PATH + file_name, 'rb')
    #   从一个打开的文件读取数据
    buffers = binfile.read()
    #   读取label文件前2个整形数字，label的长度为num
    magic, num = struct.unpack_from('>II', buffers, 0)
    #   读取labels数据
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    #   关闭文件
    binfile.close()
    #   转换为一维数组
    labels = np.reshape(labels, [num])
    return labels


x_train = load_images("train-images.idx3-ubyte")
y_train = load_labels("train-labels.idx1-ubyte")
x_test = load_images("t10k-images.idx3-ubyte")
y_test = load_labels("t10k-labels.idx1-ubyte")
train_data = (x_train, y_train)
test_data = list(zip(x_test, y_test))


def classify(point, dataSet, labels, k=3):
    """
    :param point:
    :param dataSet:
    :param labels:
    :param k:
    :return:
    """


    dataSize = dataSet.shape[0]  # 数组行数 （行，列）   # 0s

    pointArray = tile(point, (dataSize, 1))  # 扩充数组  tile(原数组, (行重复次数, 列重复次数))  # 0.057791948318481445

    subArray = pointArray - dataSet  # 数组减                                    # 0.09147500991821289

    sqrtArray = subArray * subArray  # 每个元素平方                                      # 0.08336853981018066

    """
    sum

    当axis为1时,是压缩列,即将每一行的元素相加,将矩阵压缩为一列
    当axis为0时,是压缩行,即将每一列的元素相加,将矩阵压缩为一行
    None时 所有元素相加

    """
    sumArray = sqrtArray.sum(axis=1)

    lenArray = sumArray ** 0.5  # 每个元素开方
    sortArray = lenArray.argsort()  # 按从小到大排序后返回 排序前的下标 ，eg:[300,0,400] 返回  [1, 0, 2]

    """
        unique      统计重复次数  返回值类似 ("X","Y"), (3, 1)
        zip         同步遍历多个可迭代对象  返回 ("X", 3) , ("Y", 1)
        max         求最大值，key 传一个函数，dict 传 dict.get, 多层列表可转lambda x: x[n] 返回 ("X", 3)
        [labels[index] for index in sortArray[:4]
                    求最小k个距离的样本的分类

        距离最小k个点中的分类的频率最高的
    """
    return max(zip(*unique([labels[index] for index in sortArray[:k]], return_counts=True)), key=lambda x: x[1])[0]


def test(test_data, train_data, k=3):
    index = 0
    right = 0
    start = time.time()
    for image, label in test_data:
        l = classify(image, train_data[0], train_data[1], k=3)
        if l == label:
            right += 1
        index += 1

        end = time.time()
        print("all:{index} | err: {err} err_rate:{err_rate} time:{t}".format(index=index, err=index - right,
                                                                             err_rate=(index - right) / index * 100,
                                                                             t=end - start))
    return 1.0 * (index - right) / index * 100, right, index, index - right


if __name__ == "__main__":
    err_rate, right, all_data, err = test(test_data, train_data, 3)

    print("Done! all:{all_data}|right:{right}|err:{err}".format(**vars()))

    print("err rate:{err_rate}".format(err_rate=err_rate))
