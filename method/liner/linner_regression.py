import matplotlib.pyplot as plt
import numpy as np

"""
    使用梯度下降算法实现简单的线性回归
    预测2014年南京的房价，数据集太小，并且仅仅使用了线性回归，所以并不期望其准确度，逃~
    Machine Learning 课后作业
    author 黄伟
"""


class linear_regression:
    """
        初始化函数，在这里完成超参数的定义、数据集的加载，并且调用fit()方法进行拟合
        变量说明：
            year_x, price_y 是原始数据集的年份以及相应的房价，估计单位是万元
            x, y 是经过归一化之后的数据，因为年份的值域为[2000-2014]，价格的区间为[2,13)，所以必须进行归一化，否则会爆炸
            x_mean, x_std, y_mean, y_std 归一化中产生的平均值和标准差，用于后面的预测
            lr 梯度下降的学习率
            max_iter 梯度下降最大的迭代次数
            w, b 希望回归得到的线性函数的参数
            data_size 数据集的大小，用在计算损失和偏导数中，这里为14
    """

    def __init__(self):
        self.year_x = np.array([x + 2000 for x in range(14)])
        self.price_y = np.array(
            [2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900])

        self.x, self.y, self.x_mean, self.x_std, self.y_mean, self.y_std = self.load_data()
        self.lr = 0.03
        self.max_iter = 10000
        self.w = 0
        self.b = 0
        self.data_size = len(self.x)
        self.fit()
        # self.cal()

    def cal(self):
        """
        使用最小二乘法的方式计算 系数 w 和 偏置值 b
        使用的是拓展后的 w ，将 x 即 year_x 和 全 1 矩阵在列方向上进行堆叠起来
        经过运算之后，最终得到的是一个 2 * 1 的矩阵，首个元素为 w ，其次元素为 b
        """
        X = np.vstack((self.year_x, np.ones(14))).T  # 处理堆叠后的W
        Y = self.price_y.T  # 将 price_y 进行转置，便于后面的计算
        result = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)  # 使用最小二乘法进行计算

        # ============================绘图展示======开始===============================
        plt.scatter(self.year_x, self.price_y)  # 绘制散点图
        x = np.linspace(2000, 2014, 1000)  # 给出 x 的取值散点，点多了就成线
        y = result[0] * x + result[1]  # 按照公式，计算出每个点的 predict_y
        plt.plot(x, y)  # 绘制
        plt.show()  # 展示
        # ============================绘图展示======结束===============================
        # print("the w is ", result[0], "the b is", result[1])

    """
        加载数据集的方法，在这里进行归一化
        Return 归一化的结果，以及归一化过程中的参数
    """

    def load_data(self):
        x_mean = np.mean(self.year_x)                   # 计算年份的均值
        x_std = np.std(self.year_x)                     # 计算年份的标准差
        y_mean = np.mean(self.price_y)                  # 计算房价的均值
        y_std = np.std(self.price_y)                    # 计算房价的标准差
        return (self.year_x - x_mean) / x_std, (self.price_y - y_mean) / y_std, x_mean, x_std, y_mean, y_std

    """
        拟合函数，通过计算损失函数中 w 和 b 的偏导数，进行一次次的迭代计算 w 和 b
        w = w - lr * dw
        其中 dw 表示 w 的偏导数
        最终调用show_img方法，绘制预测的图像
    """

    def fit(self):
        iter_num = 0
        while iter_num < self.max_iter:                 # 当迭代次数少于最大迭代限制时，可以进行迭代
            iter_num += 1                               # 迭代次数递增
            predict = self.w * self.x + self.b          # 计算出当前的 w 和 b 下的预测值
            loss = np.mean(sum([item ** 2 for item in (self.y - predict)]))
            # loss = (1 / self.data_size) * sum([item ** 2 for item in (self.y - predict)])
            print(f"Training: {iter_num} loop ,the loss is {loss}")
            diff_w = -(2 / self.data_size) * sum(self.x * (self.y - predict))   # 求关于 w 的偏导
            diff_b = -(2 / self.data_size) * sum(self.y - predict)              # 求关于 b 的偏导
            self.w -= self.lr * diff_w                                          # 梯度下降更新 w
            self.b -= self.lr * diff_b                                          # 梯度下降更新 b
        self.show_img()

    """
        用于数据的预测，可以输入数组和单个年份的数字，这就是使用 np 的矩阵的好处
        Return 预测的结果，和输入的类型一致
    """

    def predict(self, data):
        return (self.w * (data - self.x_mean) / self.x_std + self.b) * self.y_std + self.y_mean

    """
        绘制图像并且显示
    """

    def show_img(self):
        plt.scatter(self.year_x, self.price_y)
        x = np.linspace(2000, 2014, 1000)
        y = self.predict(x)
        plt.plot(x, y)
        # print(self.w, self.b)
        print("we think the house price of Nanjing in 2014 will be ", self.predict(2014))
        plt.show()


linear_regression()
