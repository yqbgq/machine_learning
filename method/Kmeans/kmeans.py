import numpy as np
import matplotlib.pyplot as plt
import random

x1, y1 = np.random.normal(0, 1.5, 500), np.random.normal(3, 1.5, 500)
x2, y2 = np.random.normal(-3.5, 1.5, 500), np.random.normal(-2.5, 1.5, 500)
x3, y3 = np.random.normal(4.5, 1.5, 500), np.random.normal(-2, 1.5, 500)

# 处理三类样本的坐标
data = []
for i in range(500):
    data.append([x1[i], y1[i]])
    data.append([x2[i], y2[i]])
    data.append([x3[i], y3[i]])


# 计算距离三个中心点的欧氏距离
def cal_distance(point, centers):
    return [(point[0] - x[0]) ** 2 + (point[1] - x[1]) ** 2 for x in centers]


# 决定该样本的类别，将该样本归于最近的中心点
# 返回该类别的索引
def decide_class(sample, centers):
    temp = cal_distance(sample, centers)
    return temp.index(min(temp))


# 依据每类中各个点重新计算中心点
# 返回中心点列表
def re_center(inputs, labels):
    centers = [[0.0, 0.0, 0.0] for _ in range(3)]
    for j in range(len(inputs)):
        centers[labels[j]][0] += inputs[j][0]
        centers[labels[j]][1] += inputs[j][1]
        centers[labels[j]][2] += 1.0
    centers = [[x[0] / x[2], x[1] / x[2]] for x in centers]
    return centers


# 绘制图片，包括散点图以及中心点
def show_pic(inputs, labels, centers):
    clusters = [[], [], []]
    for j in range(len(inputs)):
        clusters[labels[j]].append(data[j])
    plt.scatter([x[0] for x in clusters[0]], [x[1] for x in clusters[0]], s=30, alpha=0.7)
    plt.scatter([x[0] for x in clusters[1]], [x[1] for x in clusters[1]], s=30, alpha=0.7)
    plt.scatter([x[0] for x in clusters[2]], [x[1] for x in clusters[2]], s=30, alpha=0.7)
    plt.scatter([x[0] for x in centers], [x[1] for x in centers], linewidths=3, marker='+', s=300, c='black')
    plt.show()


# 模型训练
def fit(inputs, max_iter):
    # 随机挑选三个点作为中心点
    centers = [inputs[random.randrange(0, 1500)], inputs[random.randrange(0, 1500)], inputs[random.randrange(0, 1500)]]
    labels = [-1 for _ in range(1500)]
    iter_num = 0
    # 将样本归于最近的中心点，之后根据聚类计算新的中心点，循环直至：
    # 1. 本次循环没有改变任何一个样本的归属
    # 2. 本次循环之后，已经到达最大迭代数量
    # 新增Test
    while True:
        changed = False
        for j in range(len(inputs)):
            the_class = decide_class(inputs[j], centers)
            if the_class != labels[j]:
                changed = True
            labels[j] = the_class
        centers = re_center(inputs, labels)
        iter_num += 1
        if not changed or iter_num >= max_iter:
            break
    show_pic(inputs, labels, centers)


fit(data, 30)
