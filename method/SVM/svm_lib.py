from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt


# 绘制出分界直线
def show_pic(sets, w, b, c):
    x1, x2 = max(sets)[0], min(sets)[0]
    a1, a2 = w[0][0], w[0][1]
    # 已知w的分量a1和a2之后，可以得到：
    # a_1 x_1 + a_2 x_2 + b = 0，将x_2视为y，得到：
    # y = (-a_1 * x_1 - b) / a_2
    # 由此可以得到分界直线的两端点，进而画出相应的三条直线
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    y1, y2 = (-b - a1 * x1 + c) / a2, (-b - a1 * x2 + c) / a2
    plt.plot([x1, x2], [y1, y2])
    y1, y2 = (-b - a1 * x1 - c) / a2, (-b - a1 * x2 - c) / a2
    plt.plot([x1, x2], [y1, y2])
    plt.show()


# 定义SVM支持向量机，使用线性核，定义C为1.0
cls = svm.SVC(kernel='linear', C=1.0)
# 随机产生两个正态分布的类
x1 = np.random.normal(6, 1.5, 300)
y1 = np.random.normal(4, 1.5, 300)
x2 = np.random.normal(-1, 1.5, 300)
y2 = np.random.normal(-5, 1.5, 300)
plt.scatter(x1, y1, s=30, alpha=0.7)  # 绘制散点图
plt.scatter(x2, y2, s=30, alpha=0.7)

# 处理两类样本的坐标
data = []
label = []
for i in range(300):
    data.append([x1[i], y1[i]])
    label.append(1)
    data.append([x2[i], y2[i]])
    label.append(-1)

# 训练支持向量机
cls.fit(data, label)
w = cls.coef_           # 支持向量机的 W 向量
b = cls.intercept_      # 支持向量机的 b 偏置值
show_pic(data, w, b, 1.0)
