import matplotlib.pyplot as plt
import numpy as np
import random


# 随机选择alpha_j，要求不能和i相同
def rand_select_alpha(i, m):
    while True:
        j = random.randrange(0, m)
        if j != i:
            return j


# 如果计算alpha_j之后，超过了取值范围
# 则修剪取值
def clipAlpha(alpha_j, high, low):
    if alpha_j > high:
        alpha_j = high
    if low > alpha_j:
        alpha_j = low
    return alpha_j


# 简化版的SMO算法，其中的参数意义是：
# data - 数据矩阵
# classLabels - 数据标签
# C - 松弛变量
# error - 容错率
# maxIter - 最大迭代次数
def smo(data, labels, C, error, maxIter):
    # 转换为numpy的mat存储
    # 使用mat存储因为mat形式可以使用 * 表示矩阵相乘，而array格式只能使用.dot()函数
    # 使用multiply表示对应位置相乘
    data_matrix = np.mat(data)
    label_matrix = np.mat(labels).transpose()
    # 初始化bias参数，统计dataMatrix的维度
    bias = 0
    m, n = np.shape(data_matrix)
    # alpha_matrix，设为0
    alpha_matrix = np.mat(np.zeros((m, 1)))
    # 初始化迭代次数
    iter_num = 0
    # 最多迭代matIter次
    while iter_num < maxIter:
        change_num = 0
        for i in range(m):
            # 步骤1：计算误差Ei
            f_x_i = float(np.multiply(alpha_matrix, label_matrix).T * (data_matrix * data_matrix[i, :].T)) + bias
            e_i = f_x_i - float(label_matrix[i])
            # 优化alpha，更设定一定的容错率。
            if ((label_matrix[i] * e_i < -error) and (alpha_matrix[i] < C)) or (
                    (label_matrix[i] * e_i > error) and (alpha_matrix[i] > 0)):
                # 随机选择另一个与alpha_i成对优化的alpha_j
                j = rand_select_alpha(i, m)
                # 步骤1：计算误差Ej
                f_x_j = float(np.multiply(alpha_matrix, label_matrix).T * (data_matrix * data_matrix[j, :].T)) + bias
                e_j = f_x_j - float(label_matrix[j])
                # 保存更新前的alpha_a值，使用深拷贝
                alpha_i_old = alpha_matrix[i].copy()
                alpha_j_old = alpha_matrix[j].copy()
                # 步骤2：计算上下界L和H
                if label_matrix[i] != label_matrix[j]:
                    low = max(0, alpha_matrix[j] - alpha_matrix[i])
                    high = min(C, C + alpha_matrix[j] - alpha_matrix[i])
                else:
                    low = max(0, alpha_matrix[j] + alpha_matrix[i] - C)
                    high = min(C, alpha_matrix[j] + alpha_matrix[i])
                if low == high:
                    continue
                # 步骤3：计算eta
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T \
                      - data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    continue
                # 步骤4：更新alpha_j
                alpha_matrix[j] -= label_matrix[j] * (e_i - e_j) / eta
                # 步骤5：修剪alpha_j
                alpha_matrix[j] = clipAlpha(alpha_matrix[j], high, low)
                # 步骤6：更新alpha_i
                alpha_matrix[i] += label_matrix[j] * label_matrix[i] * (alpha_j_old - alpha_matrix[j])
                # 步骤7：更新b_1和b_2
                b1 = bias - e_i - label_matrix[i] * (alpha_matrix[i] - alpha_i_old) * data_matrix[i, :] \
                     * data_matrix[i, :].T - label_matrix[j] * (alpha_matrix[j] - alpha_j_old) \
                     * data_matrix[i, :] * data_matrix[j, :].T
                b2 = bias - e_j - label_matrix[i] * (alpha_matrix[i] - alpha_i_old) * data_matrix[i, :] \
                     * data_matrix[j, :].T - label_matrix[j] * (alpha_matrix[j] - alpha_j_old) \
                     * data_matrix[j, :] * data_matrix[j, :].T
                # 步骤8：根据b_1和b_2更新b
                if (0 < alpha_matrix[i]) and (C > alpha_matrix[i]):
                    bias = b1
                elif (0 < alpha_matrix[j]) and (C > alpha_matrix[j]):
                    bias = b2
                else:
                    bias = (b1 + b2) / 2.0
                # 统计优化次数
                change_num += 1
        # 更新迭代次数,如果有过更新，则将迭代数量清零，否则迭代数量递增
        if change_num == 0:
            iter_num += 1
        else:
            iter_num = 0
    return bias, alpha_matrix


# 将分类结果可视化
# dataMat 表示数据矩阵
# w 是法向量
# b是原点到直线的距离
def show_pic(sets, w, b, c):
    # 绘制直线
    x1 = max(sets)[0]
    x2 = min(sets)[0]
    # 这里从matrix中取出来是[[e]]的形式
    a1, a2 = w
    b = float(b)
    # 因为取出来还是[[e]]的格式，所以进行转化为浮点数
    a1 = float(a1)
    a2 = float(a2)
    # 已知w的分量a1和a2之后，可以得到：
    # a_1 x_1 + a_2 x_2 + b = 0，将x_2视为y，得到：
    # y = (-a_1 * x_1 - b) / a_2
    # 由此可以得到分界直线的两端点，进而画出直线来
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    y1, y2 = (-b - a1 * x1 + c) / a2, (-b - a1 * x2 + c) / a2
    plt.plot([x1, x2], [y1, y2])
    y1, y2 = (-b - a1 * x1 - c) / a2, (-b - a1 * x2 - c) / a2
    plt.plot([x1, x2], [y1, y2])
    plt.show()


# 计算出w
def get_w(data, label, alpha):
    # w是 sum(alpha_i * y_i * x_i.T)
    weight = []
    for x in range(len(data[0])):
        temp = 0.0
        for i in range(len(data)):
            temp += alpha[i] * label[i] * data[i][x]
        weight.append(temp)
    return weight


if __name__ == '__main__':
    x1 = np.random.normal(6, 1.5, 100)
    y1 = np.random.normal(4, 1.5, 100)

    x2 = np.random.normal(-1, 1.5, 100)
    y2 = np.random.normal(-5, 1.5, 100)
    plt.scatter(x1, y1, s=30, alpha=0.7)  # 正样本散点图
    plt.scatter(x2, y2, s=30, alpha=0.7)  # 负样本散点图

    data = []
    label = []
    for i in range(100):
        data.append([x1[i], y1[i]])
        label.append(1)
        data.append([x2[i], y2[i]])
        label.append(-1)
    b, alphas = smo(data, label, 0.6, 0.001, 40)
    w = get_w(data, label, alphas)
    show_pic(data, w, b, 0.6)
