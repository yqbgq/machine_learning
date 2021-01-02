import numpy as np
import matplotlib.pyplot as plt
from typing import List


class isodata:
    def __init__(self, k, theta_n, theta_s, theta_c, max_combine_num, max_iter):
        """
        isodata 类的初始化函数，将创建实例时传入的参数写入类内

        :param k: 期望聚类的数量
        :param theta_n: 每个聚类中最少的样本个数
        :param theta_s: 分裂参数
        :param theta_c: 合并参数
        :param max_combine_num: 每次迭代最多合并次数
        :param max_iter: 迭代上限
        """
        self.except_k = k
        self.theta_n = theta_n
        self.theta_s = theta_s
        self.theta_c = theta_c
        self.max_combine_num = max_combine_num
        self.max_iter = max_iter

    def train(self, debug=False):
        """
        训练，使用 debug 参数控制是否在最后输出关键信息

        :param debug: debug 开关
        """
        self.__get_data()  # 输入数据
        self.__init_center()  # 初始化，设置最初的聚类中心，设置初始聚类数量为 except_k / 2
        for x in range(self.max_iter):
            temp_max_combine_num = self.max_combine_num  # 保存迭代中最多合并次数
            self.__decide_class()  # 将样本归类于某个聚类中心
            self.__get_clusters()  # 将样本划分为各个不同的簇
            self.__remove_class()  # 如若某个聚类中的样本数量少于 theta_n，那么删除这个类，重新安排聚类中心
            self.__cal_avg()  # 计算各种均值
            if x != self.max_iter - 1:  # 如若不是最后一次迭代，则要进行决定是否要进行分裂或者合并
                self.__split_or_combine(x)
            self.max_combine_num = temp_max_combine_num  # 重置迭代中最多合并次数
        if debug:  # 如果开启了 debug
            self.__print_info()  # 输出信息
        self.__show_pic()  # 展示聚类结果

    def __get_data(self):
        """
        为程序设置测试数据
        """
        x1, y1 = np.random.normal(-1.5, 1.5, 500), np.random.normal(5, 1.5, 500)
        x2, y2 = np.random.normal(-5.5, 1.5, 500), np.random.normal(-2.5, 1.5, 500)
        x3, y3 = np.random.normal(3.5, 1.5, 500), np.random.normal(-3, 1.5, 500)
        # 处理三类样本的坐标
        self.data = []
        for i in range(500):
            self.data.append([x1[i], y1[i]])
            self.data.append([x2[i], y2[i]])
            self.data.append([x3[i], y3[i]])
        self.data = np.asarray(self.data)  # type: np.ndarray

    def __init_center(self):
        """
        初始化中心点，一开始随机选择 k // 2个聚类中心
        """
        self.current_class = self.except_k // 2
        # 随机选择不重复的 current_class 个聚类中心
        indexes = np.random.choice(range(len(self.data)), self.current_class, replace=False)
        # 设置聚类中心
        self.centers = self.data[indexes]  # type: np.ndarray

    def __decide_class(self):
        """计算每个样本到达各个聚类中心的距离，获得每个样本的所属类别 result"""
        temp_data_0 = np.expand_dims(self.data, len(self.data.shape) - 1).repeat(len(self.centers),
                                                                                 len(self.data.shape) - 1)
        temp_data_1 = temp_data_0 - self.centers
        temp_data_2 = temp_data_1 ** 2
        temp_data_3 = np.sum(temp_data_2, axis=2)
        self.result = np.argmin(temp_data_3, axis=1)

    def __get_clusters(self):
        """按照 __decide_class 方法获得的 result 将样本分为所属的簇中"""
        indexes = []
        self.clusters = []  # type: List[np.ndarray]
        for x in range(self.current_class):
            _ = np.argwhere(self.result == x)
            indexes.append(_.reshape(_.shape[0]))
        for x in indexes:
            self.clusters.append(self.data[x])
        self.current_class = len(self.clusters)
        self.__recenter()

    def __remove_class(self):
        """删除类内样本数量太少的聚类，同时重新进行聚类中心以及簇的分配"""
        retain_indexes = []
        # 检查各个聚类中样本数量，大于 theta_n 数量的簇，将其编号加入 retain_indexes 中
        for idx in range(self.current_class):
            if self.clusters[idx].shape[0] > self.theta_n:
                retain_indexes.append(idx)
        # 获取要被去除的编号
        removing_indexes = [x for x in range(self.current_class) if x not in retain_indexes]
        # 如若要删除合并的聚类数量大于上限，则从要删除的编号中取出一部分，以满足合并上限的限制
        if len(removing_indexes) > self.max_combine_num:  # 确保合并的数量不大于最大合并数量
            retain_indexes.extend(removing_indexes[self.max_combine_num:])
            self.max_combine_num = 0
        # 如果 retain_indexes 中的数量少于 current_class，说明需要进行合并
        if len(retain_indexes) < self.current_class:
            # 重新获取聚类中心
            self.centers = self.centers[retain_indexes]  # type: np.ndarray
            # 更新 current_class，即当前拥有的聚类数目
            self.current_class = len(retain_indexes)
            # 重新分配聚类中心和簇
            self.__decide_class()
            self.__get_clusters()

    def __recenter(self):
        """重新计算聚类中心"""
        for i in range(self.current_class):
            self.centers[i] = np.mean(self.clusters[i], axis=0)

    def __cal_avg(self):
        """计算各种平均值"""
        self.avg = []  # type: List[np.ndarray]
        # 计算各个簇中的平均值
        for x in self.clusters:
            self.avg.append(np.mean(x, axis=0))
        self.avg_dis = []
        self.total_avg_dis = 0
        # 计算各个簇中，样本点到簇中心的距离的平均值
        # 以及总平均距离
        for i in range(self.current_class):
            dis = np.mean(np.sum((self.clusters[i] - self.avg[i]) ** 2, axis=1) ** (1 / 2))
            self.avg_dis.append(dis)
            self.total_avg_dis += dis * self.clusters[i].shape[0]
        self.total_avg_dis /= self.data.shape[0]

    def __split_or_combine(self, iter_num):
        """判断是分裂还是合并"""
        # 如若当前的聚类数量大于等于 k * 2 或者当前迭代次数为偶数次，则进行合并
        if self.current_class >= self.except_k * 2 or iter_num % 2 == 0:
            self.__combine()
        elif self.current_class < self.except_k // 2:  # 如若当前的聚类数量小于等于 k/2，则进行分裂
            self.__split(k=0.6)

    def __split(self, k):
        """进行分裂， k 是分裂后两个聚类中心的计算参数，差值为 ± k * 各维标准差"""
        split_num = 0  # 已分裂的数量
        deleting_class = []  # 需要删除的聚类中心编号
        for i in range(self.current_class):
            # 计算各维标准差
            std = np.mean((self.clusters[i] - self.avg[i]) ** 2, axis=0) ** (1 / 2)
            # 获得最大标准差
            theta_max = np.max(std)
            # 如若最大标准差大于 theta_s，且该簇中样本到中心的平均距离大于总平均距离，且簇中样本数或者聚类个数满足一定要求，进行分裂
            if (theta_max > self.theta_s and self.avg_dis[i] > self.total_avg_dis
                and self.clusters[i].shape[0] > 2 * (self.theta_n + 1)) or (self.current_class + split_num)\
                    <= self.except_k // 2:
                split_num += 1
                self.centers = np.row_stack([self.centers, self.centers[i] + k * std])  # 分裂，新增聚类中心
                self.centers = np.row_stack([self.centers, self.centers[i] - k * std])
                deleting_class.append(i)
        # 去除需要被删除的聚类中心，最后才删除，不然会出现错误
        retain_indexes = [x for x in range(self.centers.shape[0]) if x not in deleting_class]
        self.centers = self.centers[retain_indexes]
        # 更新聚类数目
        self.current_class += split_num

    def __combine(self):
        """进行合并"""
        dis_between_class = []
        # 如果还有可以合并的余量
        if self.max_combine_num > 0:
            # 计算各个聚类中心之间的距离，这里如果用矩阵来算的话可以更快
            # 但是不知道怎么整
            for i in range(self.current_class):
                for j in range(self.current_class):
                    if i > j:  # 和非自身且不重复的距离
                        dis = np.sum((self.centers[i] - self.centers[j]) ** 2) ** (1 / 2)
                        # 距离小于阈值 theta_c
                        if dis < self.theta_c:
                            # 距离，聚类中心1编号，聚类中心2编号
                            dis_between_class.append([dis, i, j])
        combined_class = []
        # 按照距离进行排序，并且进行反转，方便后面的 pop 弹出
        dis_between_class = sorted(dis_between_class, key=lambda e: e[0], reverse=True)
        deleting_center = []
        while self.max_combine_num > 0 and len(dis_between_class) > 0:
            # 弹出一个可用的 item
            item = dis_between_class.pop()
            i = item[1]
            j = item[2]
            # 防止一个聚类中心被重复合并
            if i not in combined_class and j not in combined_class:
                self.max_combine_num -= 1
                deleting_center.append(i)
                deleting_center.append(j)
                num_of_i = self.clusters[i].shape[0]
                num_of_j = self.clusters[j].shape[0]
                # 计算新的合并后的聚类中心，并且加入到聚类中心列表中
                new_center = (num_of_i * self.avg[i] + num_of_j * self.avg[j]) / (num_of_i + num_of_j)
                self.centers = np.row_stack([self.centers, new_center])
        # 删除被合并了的聚类中心
        retain_indexes = [x for x in range(self.centers.shape[0]) if x not in deleting_center]
        self.centers = self.centers[retain_indexes]
        self.current_class = self.centers.shape[0]

    def __show_pic(self):
        """展示聚类后的图片"""
        for cluster in self.clusters:
            plt.scatter([x[0] for x in cluster], [x[1] for x in cluster], s=30, alpha=0.7)
        for center in self.centers:
            plt.scatter(center[0], center[1], linewidths=3, marker='+', s=300, c='black')
        plt.show()

    def __print_info(self):
        """输出 debug 信息"""
        print(self.avg)
        print(self.centers)
        for i in range(self.current_class):
            std = np.mean((self.clusters[i] - self.avg[i]) ** 2, axis=0) ** (1 / 2)
            print(std)
        if self.max_combine_num > 0:
            for i in range(self.current_class):
                for j in range(self.current_class):
                    if i > j:
                        dis = np.sum((self.centers[i] - self.centers[j]) ** 2) ** (1 / 2)
                        print(dis)


if __name__ == "__main__":
    # 创建一个 isodata 的实例，参数含义依次为：
    # 预期聚类数量，每个聚类中最少的聚类数量
    # 分裂参数，聚合参数
    # 每次迭代最多合并多少次，迭代次数上限
    instance = isodata(9, 80, 5.0, 0.01, 5, 100)
    instance.train(debug=False)  # 开始训练
