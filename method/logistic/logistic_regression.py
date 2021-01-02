import matplotlib.pyplot as plt
import numpy as np
import random

# 数据文件的存储路径  the path of data file
path_x = "x.txt"
path_y = "y.txt"
path_x_test = "x.txt"
path_y_test = "y.txt"
# 为显示中文，添加字体  add front to show chinese in fig
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


def get_scatter(x_path: str, y_path: str) -> tuple:
    """
    读取数据文件，按照标签分类，返回 x 和 y 的坐标，用于绘制散点图
    Read the data file, sort by label,  return the coordinates of x and y which are used to draw scatter plot
    :param x_path: x坐标文件路径
    :param y_path: y坐标文件路径
    :return: 按照标签，返回正负类的 x 坐标和 y 坐标， 点坐标以及相应的标签值
    """
    with open(x_path) as f:
        dataset = f.read()
    # 处理从文件中读取的数据，进行清洗 clean the data read from file
    data = [float(x) for x in dataset.replace("\n", " ").split(" ")[:-1]]
    num = int(len(data) / 2)
    data_x = [data[2 * i] for i in range(num)]
    data_y = [data[2 * i + 1] for i in range(num)]
    points = [[data[2 * i], data[2 * i + 1]] for i in range(num)]
    with open(y_path) as f:
        dataset = f.read()
    labels = [float(x) for x in dataset.split("\n")[:-1]]
    # 使用推导式，进行将坐标数据进行分类 Using the derivation, classify coordinate data
    x1 = [data_x[i] for i in range(num) if labels[i] == 0.0]
    x2 = [data_x[i] for i in range(num) if labels[i] == 1.0]
    y1 = [data_y[i] for i in range(num) if labels[i] == 0.0]
    y2 = [data_y[i] for i in range(num) if labels[i] == 1.0]
    return x1, y1, x2, y2, np.array(points), np.array(labels)


# 用于处理逻辑回归的类
# Class for handling logical regression
class logistic:
    def __init__(self):
        # 获取散点，用于后续画图 get scatters which are used to plot
        self.x1, self.y1, self.x2, self.y2, self.dataset, self.labels = get_scatter(path_x, path_y)
        _, _, _, _, self.point_test, self.labels_test = get_scatter(path_x_test, path_y_test)
        # 改变标签的形状，便于后续计算 Change the shape of the label to facilitate subsequent calculation
        self.labels = self.labels.reshape([-1, 1])
        # 对数据进行预处理，返回平均值和标准差 preprocess the data, return mean and std
        self.mean, self.std = self._pre_process()
        # 学习率 learning rate
        self.lr = 0.2
        self.iter_max = 200  # 最大迭代次数
        # theta 和损失的列表，用于后续的画图 the lists of theta and loss which are used to plot the process
        self.theta_list = []
        self.loss_list = []
        self.error_num_list = []

    def _pre_process(self) -> tuple:
        """
        对数据进行预处理，返回数据的平均值和标准差
        :return: 平均值 标准差
        """
        mean = np.mean(self.dataset, axis=0)
        std = np.std(self.dataset, axis=0)
        self.dataset = (self.dataset - mean) / std
        self.dataset = np.hstack((self.dataset, np.ones([len(self.dataset), 1])))
        return mean, std

    def gd(self, sgd=False):
        """
        实现梯度下降或者随机梯度下降的算法
        :param sgd: False ： 使用梯度下降   True ： 使用随机梯度下降
        """
        count = 0  # 当前已经迭代次数 Current number of iter
        theta = np.ones([3, 1])  # 初始化theta， 分别代表 w1, w2, b init the theta, denote w1, w2, b
        while count < self.iter_max:
            if sgd:  # 如果当前是随机梯度下降的话    if use sgd method
                idx = random.randrange(0, 64)  # 随机选择一个样本进行梯度下降  choose a sample to sgd randomly
                h_theta_x = 1 / (1 + np.exp(-np.dot(self.dataset[idx], theta)))  # 计算hθ(x)  cal hθ(x)
                cost = np.reshape((self.labels[idx] - h_theta_x) * self.dataset[idx], [-1, 1])
                theta = theta + self.lr * cost  # 更新theta renew theta
            else:
                h_theta_x = 1 / (1 + np.exp(-np.dot(self.dataset, theta)))
                cost = np.mean((self.labels - h_theta_x) * self.dataset, axis=0).reshape([-1, 1])
                theta = theta + self.lr * cost  # 按照公式更新theta renew the theta according to equation
            count += 1  # 新增迭代次数
            self._record(theta)
        self._show()  # 绘图 plot

    def newTon(self):
        """
        实现牛顿法的函数， the func of newTon's method
        """
        theta = np.ones([3, 1])  # 初始化 init
        count = 0  # 记录迭代次数 the num of iter
        while count < self.iter_max:
            h_theta_x = 1.0 / (1 + np.exp(-np.dot(self.dataset, theta)))  # 计算hθ(x)
            d_theta = (self.labels - h_theta_x) * self.dataset  # 计算一阶导数
            d_theta_2 = np.zeros((3, 3))  # 初始化二阶导数
            for i in range(len(self.dataset)):  # 按照公式，计算二阶导数
                x = np.array(self.dataset[i]).reshape([3, 1])
                h_theta_x = 1.0 / (1 + np.exp(-np.dot(x.T, theta)))
                d_theta_2 += h_theta_x[0, 0] * (h_theta_x[0, 0] - 1) * np.dot(x, x.T)
            theta = theta - np.dot(np.linalg.inv(d_theta_2), np.sum(d_theta, axis=0).reshape([-1, 1]))  # 更新迭代
            count += 1
            self._record(theta)
        self._show()

    def _predict(self, theta: np.ndarray, dataset: np.ndarray) -> float:
        """
        按照给定的theta， 预测 dataset 对应的 y 值
        :param theta: 使用的 theta
        :param dataset: 需要预测的 x
        :return: 返回预测结果
        """
        return (-(theta[0] * ((dataset - self.mean[0]) / self.std[0]) + theta[2]) / theta[1]) * self.std[1] + self.mean[
            1]

    def _cal_loss(self, theta: np.ndarray) -> float:
        """
        在训练数据集上，计算当前 theta 对应的损失
        :param theta:  当前使用的 theta
        :return: 返回当前的 theta 下的损失
        """
        h_theta_x = 1 / (1 + np.exp(-np.dot(self.dataset, theta)))
        loss = self.labels * np.log10(h_theta_x) + (1 - self.labels) * np.log10(1 - h_theta_x)
        return abs(np.mean(loss))

    def _error_num(self, theta: np.ndarray):
        """
        检查在测试数据集上，误分类样本的个数，添加到列表中用于绘图
        :param theta: 当前程序训练得到的 θ
        :return: None
        """
        error_num = 0
        for i in range(len(self.point_test)):
            x = self.point_test[i]
            data_after_process = [(x[i] - self.mean[i]) / self.std[i] for i in range(2)]
            idx = data_after_process[0] * theta[0] + data_after_process[1] * theta[1] + theta[2]
            out = 1 / (1 + np.exp(-idx))
            if self.labels_test[i] == 1:
                if out[0] < 0.5:
                    error_num += 1
            else:
                if out[0] > 0.5:
                    error_num += 1
        self.error_num_list.append(error_num)

    def _record(self, theta: np.ndarray):
        """
        记录当前迭代的信息，包括：当前的 θ， 在当前 θ 下训练集上的损失函数值以及在当前的 θ 下，测试集上误分的数量
        :param theta:  θ
        :return:   None
        """
        self.theta_list.append(theta)
        self.loss_list.append(np.mean(self._cal_loss(theta)))
        self._error_num(theta)

    def _show(self):
        """
        遍历 theta 的历史记录和 loss 的历史记录，绘图展示
        """
        for i in range(len(self.loss_list)):
            plt.clf()
            plt.subplot(1, 3, 1)
            plt.title("分类可视化")
            plt.scatter(self.x1, self.y1, s=30, alpha=0.7)  # 正样本散点图
            plt.scatter(self.x2, self.y2, s=30, alpha=0.7)  # 正样本散点图
            x = np.array([20, 55])
            y = self._predict(self.theta_list[i], x)
            plt.plot(x, y)
            plt.subplot(1, 3, 2)
            plt.title("损失函数图")
            plt.scatter(range(i), self.loss_list[:i], s=30, alpha=0.7)
            plt.subplot(1, 3, 3)
            plt.title("测试集上分类错误数")
            plt.scatter(range(i), self.error_num_list[:i], s=30, alpha=0.7)
            if i != len(self.loss_list) - 1:
                plt.pause(0.0001)
            else:
                plt.show()


if __name__ == "__main__":
    logistic_regression = logistic()
    # logistic_regression.newTon()  # 使用牛顿法进行逻辑回归
    # logistic_regression.gd()  # 使用随机梯度下降方法进行逻辑回归
    logistic_regression.gd(sgd=True)  # 使用梯度下降算法进行逻辑回归
