import numpy as np
import matplotlib.pyplot as plt

# 训练数据存储位置
path_x = "train/x.txt"
path_y = "train/y.txt"

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


class layer:
    def __init__(self, in_dim, out_dim, lr):
        """
        全连接层的初始化
        :param in_dim:  输入的向量大小
        :param out_dim: 输出的向量大小
        :param lr:      学习率
        """
        self.out_dim = out_dim
        self.w = np.random.random([in_dim, out_dim])  # 随机初始化权重矩阵
        self.delta = None
        self.out = None
        self.lr = lr
        self.in_flow = None

    def forward(self, in_flow: np.ndarray):
        """
        前向传播
        :param in_flow: 输入向量
        """
        self.in_flow = in_flow  # 前向传播的输入
        self.out = np.dot(in_flow, self.w)  # 通过权重计算输出
        self.out = 1 / (1 + np.exp(-self.out))  # Sigmoid 输出
        return self.out

    def set_delta(self, before):
        """
        设置梯度
        :param before: 梯度
        """
        self.delta = before

    def gd(self):
        """
        进行梯度下降
        """
        self.w -= self.lr * np.dot(self.in_flow.T, self.delta * self.out * (1 - self.out))


class ann:
    def __init__(self, lr, max_iter):
        self.points, self.labels = get_data()  # 读取样本点以及标签信息
        self.avg, self.std = 0, 0  # 定义样本的平均值以及
        self.iter = max_iter

        # ============================ 分别划分三类的 X 和 Y 轴的列表==================================
        self.class_one_x = [self.points[x][0] for x in range(120) if self.labels[x][0] == 1]
        self.class_one_y = [self.points[x][1] for x in range(120) if self.labels[x][0] == 1]
        self.class_two_x = [self.points[x][0] for x in range(120) if self.labels[x][1] == 1]
        self.class_two_y = [self.points[x][1] for x in range(120) if self.labels[x][1] == 1]
        self.class_three_x = [self.points[x][0] for x in range(120) if self.labels[x][2] == 1]
        self.class_three_y = [self.points[x][1] for x in range(120) if self.labels[x][2] == 1]
        # ============================ 分别划分三类的 X 和 Y 轴的列表==================================

        self.__normal()  # 将样本归一化，否则可能会不收敛
        self.points = np.hstack((self.points, np.ones([len(self.points), 1])))  # 添加一个特征为 1，用于写偏置值
        self.nn1 = layer(3, 5, lr)  # 构造全连接层 1
        self.nn2 = layer(5, 4, lr)  # 构造全连接层 2
        self.nn3 = layer(4, 3, lr)  # 构造全连接层 2

    def __forward(self):
        """
        前向传播
        """
        self.nn1.forward(self.points)
        self.nn2.forward(self.nn1.out)
        self.nn3.forward(self.nn2.out)

    def __backward(self):
        """
        反向传播
        """
        delta3 = self.nn3.out - self.labels  # 计算梯度
        delta2 = (delta3 * self.nn3.out * (1 - self.nn3.out)).dot(self.nn3.w.T)
        delta1 = (delta2 * self.nn2.out * (1 - self.nn2.out)).dot(self.nn2.w.T)
        self.nn3.set_delta(delta3)
        self.nn2.set_delta(delta2)
        self.nn1.set_delta(delta1)

    def __gd(self):
        """
        计算梯度下降
        """
        self.nn3.gd()
        self.nn2.gd()
        self.nn1.gd()

    def train(self):
        """
        进行训练
        """
        for i in range(self.iter):
            self.__forward()
            self.__backward()
            self.__gd()
            self.__check_the_loss(i)  # 检查损失
        self.__show()  # 展示结果

    def __check_the_loss(self, step):
        """
        计算当前网络的损失
        :param step: 迭代次数
        """
        self.nn1.forward(self.points)
        self.nn2.forward(self.nn1.out)
        self.nn3.forward(self.nn2.out)
        result = self.nn3.out
        loss = 0.5 * np.sum((self.labels - result) * (self.labels - result))

        print("step {} the loss is {}".format(step, loss))

    def __normal(self):
        """
        样本点的归一化，使用的方法是： (样本 - 均值) / 方差
        """
        self.avg = np.mean(self.points, axis=0).reshape([-1, 2])
        self.std = np.std(self.points, axis=0).reshape([-1, 2])
        self.points = (self.points - self.avg) / self.std

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

        self.nn1.forward(point)
        self.nn2.forward(self.nn1.out)
        self.nn3.forward(self.nn2.out)
        result = self.nn3.out

        return np.argmax(result, axis=1)  # 使用 argmax 返回类别

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

        x = np.linspace(2, 4.5, 200)  # x 的列表
        y = np.linspace(0, 2.5, 200)  # y 的列表
        X, Y = np.meshgrid(x, y)  # 进行组合，得到需要预测的点

        for i in range(200):
            for j in range(200):
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


# 构造 ANN 实例，学习率为 0.2， 迭代次数为 10000
a = ann(0.2, 10000)
# 开始训练，因为参数是随机初始化的，而且神经网络可以拟合任意函数，所以可能会画出很过拟合的图像
a.train()
