from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 使用正态分布随机产生三个类别的样本坐标
x1, y1 = np.random.normal(0, 1.5, 500), np.random.normal(3, 1.5, 500)
x2, y2 = np.random.normal(-3.5, 1.5, 500), np.random.normal(-2.5, 1.5, 500)
x3, y3 = np.random.normal(4.5, 1.5, 500), np.random.normal(-2, 1.5, 500)

# 处理两类样本的坐标
data = []
for i in range(500):
    data.append([x1[i], y1[i]])
    data.append([x2[i], y2[i]])
    data.append([x3[i], y3[i]])

km = KMeans(n_clusters=3).fit(data)
clusters = [[], [], []]
center1, center2, center3 = [], [], []

for i in range(len(data)):
    clusters[km.labels_[i]].append(data[i])

center1 = [sum([x[0] for x in clusters[0]]) / len(clusters[0]), sum([x[1] for x in clusters[0]]) / len(clusters[0])]
center2 = [sum([x[0] for x in clusters[1]]) / len(clusters[1]), sum([x[1] for x in clusters[1]]) / len(clusters[1])]
center3 = [sum([x[0] for x in clusters[2]]) / len(clusters[2]), sum([x[1] for x in clusters[2]]) / len(clusters[2])]

plt.scatter([x[0] for x in clusters[0]], [x[1] for x in clusters[0]], s=30, alpha=0.7)
plt.scatter([x[0] for x in clusters[1]], [x[1] for x in clusters[1]], s=30, alpha=0.7)
plt.scatter([x[0] for x in clusters[2]], [x[1] for x in clusters[2]], s=30, alpha=0.7)
plt.scatter(center1[0], center1[1], linewidths=3, marker='+', s=300, c='black')
plt.scatter(center2[0], center2[1], linewidths=3, marker='+', s=300, c='black')
plt.scatter(center3[0], center3[1], linewidths=3, marker='+', s=300, c='black')
plt.show()
