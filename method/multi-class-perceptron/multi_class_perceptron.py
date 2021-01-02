import numpy as np
import matplotlib.pyplot as plt

# 训练数据存储位置
path_x = "../logistic/x.txt"
path_y = "../logistic/y.txt"

# 为显示中文，添加字体  add front to show chinese in fig
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


def get_data() -> (np.array, np.array):
    """
    从文件中读取数据，返回的是两个矩阵
        1. 点坐标矩阵，大小为 150 X 2
        2. 标签矩阵，已经转化为了 one hot 编码，方便后面的计算
    :return:
        样本坐标矩阵以及标签的 one hot 编码
    """
    with open(path_x) as f:
        temp = f.read().split("\n")  # 读取并且分割、处理
        points = [[float(x.split(" ")[0]), float(x.split(" ")[1])] for x in temp if len(x) > 0]
    with open(path_y) as f:
        temp = f.read().split("\n")  # 读取并且分割、处理
        labels = [int(x) for x in temp if len(x) > 0]
    return np.array(points), np.eye(3)[labels]  # 返回点的左边，和标签的 one hot 编码


class m_c_perceptron:
    """
    多分类感知机
    """
    def __init__(self):
        self.points, self.labels = get_data()  # 读取样本点以及标签信息
        self.avg, self.std = 0, 0  # 定义样本的平均值以及

        # ============================ 分别划分三类的 X 和 Y 轴的列表==================================
        self.class_one_x = [self.points[x][0] for x in range(120) if self.labels[x][0] == 1]
        self.class_one_y = [self.points[x][1] for x in range(120) if self.labels[x][0] == 1]
        self.class_two_x = [self.points[x][0] for x in range(120) if self.labels[x][1] == 1]
        self.class_two_y = [self.points[x][1] for x in range(120) if self.labels[x][1] == 1]
        self.class_three_x = [self.points[x][0] for x in range(120) if self.labels[x][2] == 1]
        self.class_three_y = [self.points[x][1] for x in range(120) if self.labels[x][2] == 1]
        # ============================ 分别划分三类的 X 和 Y 轴的列表==================================

        self.__normal()  # 将样本归一化，否则可能会不收敛

        self.points = np.hstack((self.points, np.ones([len(self.points), 1])))  # 添加一列全为 1 的，用于处理偏置值 150 X 3
        self.w = np.ones([3, 3])                                                # 权重，添加了一行 1 ，用于处理偏置值
        self.__gd()                                                             # 梯度下降
        self.__show()                                                           # 展示

    def __normal(self):
        """
        样本点的归一化，使用的方法是： (样本 - 均值) / 方差
        """
        self.avg = np.mean(self.points, axis=0).reshape([-1, 2])
        self.std = np.std(self.points, axis=0).reshape([-1, 2])
        self.points = (self.points - self.avg) / self.std

    def __gd(self):
        for x in range(10000):
            predict = self.points.dot(self.w)           # 预测结果
            # ============= 将预测结果转化为： 每一行中最大的值为 1 ，其余为 0 形式的矩阵 =================
            index = np.argsort(predict).T[2]
            max_index = np.eye(3)[index].astype(bool)
            predict[max_index] = 1
            predict[~max_index] = 0
            # ============= 将预测结果转化为： 每一行中最大的值为 1 ，其余为 0 形式的矩阵 =================
            gradient = self.points.T.dot(predict - self.labels)  # 按照迭代公式进行迭代，更新权值
            self.w -= 0.0001 * gradient                             # 更新

    def __show(self):
        """
        展示训练之后的 theta 得到的划分可视化图像
        """
        # =============== 基本的设置 ============================
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.title("分类可视化")
        colors = ["yellow", "pink", "blue"]
        samples = []  # 样本点列表
        # =============== 基本的设置 ============================

        x = np.linspace(2, 4.5, 150)  # x 的列表
        y = np.linspace(0, 2.5, 150)  # y 的列表
        X, Y = np.meshgrid(x, y)  # 进行组合，得到需要预测的点

        for i in range(150):
            for j in range(150):
                samples.append([X[i][j], Y[i][j]])  # 加入需要预测的样本点

        result = self.__predict(np.mat(samples)).reshape([-1, 1])  # 处理预测，得到结果

        color = [colors[x[0]] for x in result.tolist()]  # 处理每个预测点的结果，点够多就可以近似看做背景被渲染了

        # ==================== 渲染背景，绘制训练样本点=========================================
        plt.scatter(X, Y, color=color, s=20)
        plt.scatter(self.class_one_x, self.class_one_y, s=20, color="red", marker='x')
        plt.scatter(self.class_two_x, self.class_two_y, s=20, color="green", marker='s')
        plt.scatter(self.class_three_x, self.class_three_y, s=20, color="gray", marker="*")
        # ==================== 渲染背景，绘制训练样本点=========================================

        plt.show()  # 展示图像

    def __predict(self, point: np.mat) -> np.array:
        """
        输入样本点矩阵，输出每个样本点对应的类别

        :param point:
            样本点矩阵
        :return:
            类型矩阵
        """
        point = (point - self.avg) / self.std  # 对样本点矩阵进行正则化处理，统一在转换后的空间后进行运算
        point = np.hstack((point, np.ones([point.shape[0], 1])))  # 按照对训练点的处理方法，在最后附加一列的 1，用于处理偏置值
        sums = np.sum(np.exp(np.dot(point, self.w)))
        h_x = np.exp(np.dot(point, self.w)) / sums  # 处理计算 SoftMax 的值
        return np.argmax(h_x, axis=1)  # 使用 argmax 返回类别


perceptron = m_c_perceptron()
