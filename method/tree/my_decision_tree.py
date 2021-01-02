from math import log


# 定义决策树节点
class node:
    # 节点的构造函数
    def __init__(self):
        self.node_class = ""  # 节点的类型，分为 “decision” 判断节点以及 "end" 叶子节点
        self.name = ""  # 对于判断节点，该属性为使用的特征名，对于叶子节点，该属性表示最终分类结果
        self.next = []  # 子节点列表，对于判断节点而言，表示使用该特征进行分类后对应生成的下一步节点。若为叶子节点，则该列表为空
        self.feature = -1  # 表示叶子节点使用了哪个特征，该属性是该特征在特征列表上的索引


def createDataSet():
    """
    数据集来自《模式识别》清华大学出版社
    特征1： 年龄  0: 小于30 1：大于30
    特征2： 性别  0： 男    1： 女
    特征3： 薪水  0： 低    1： 中    2： 高
    :return: 返回数据集以及标签集合
    """
    datasets = [[0, 0, 1, '否'],  # 数据集
                [1, 1, 1, '否'],
                [1, 1, 1, '否'],
                [1, 1, 0, '否'],
                [0, 0, 2, '否'],
                [1, 1, 0, '否'],
                [0, 1, 0, '否'],
                [0, 1, 2, '是'],
                [1, 0, 1, '是'],
                [0, 0, 2, '否'],
                [1, 1, 1, '否'],
                [0, 0, 0, '否'],
                [1, 1, 1, '否'],
                [1, 0, 0, '是'],
                [1, 0, 1, '是'],
                [1, 1, 0, '否']]
    label_list = ['年龄', '性别', '月收入']  # 特征标签
    return datasets, label_list  # 返回数据集和分类属性


def class_data_by_class(datasets):
    """
    将数据集中的数据按照类别进行分类，返回字典，键是类别，值是数据列表
    :param datasets: 未分类的数据集
    :return: 根据类别进行分类后的数据集字典
    """
    class_map = {}
    for x in datasets:
        if x[-1] not in class_map.keys():
            class_map[x[-1]] = [x]
        else:
            class_map[x[-1]].append(x)
    return class_map


def class_data_by_feature(datasets, feature):
    """
    将数据集中的数据按照特征进行分类，返回字典，键是在特征上的不同取值，值是数据列表
    :param datasets:
    :param feature:
    :return:根据特征上的不同取值进行分类后的数据集字典
    """
    feature_map = {}
    for x in datasets:
        if x[feature] not in feature_map.keys():
            feature_map[x[feature]] = [x]
        else:
            feature_map[x[feature]].append(x)
    return feature_map


def cal_empirical_entropy(class_map, data_size):
    """
    计算经验熵 ent_d
    :param class_map: 按照类别分好的数据字典
    :param data_size: 总数据集的大小
    :return: 经验熵
    """
    ent_d = 0
    for x in list(class_map.keys()):
        ent_d -= len(class_map[x]) / data_size * log(len(class_map[x]) / data_size, 2)
    return ent_d


def cal_conditional_entropy(datasets, feature, data_size):
    """
    计算条件熵 ent_a
    :param datasets: 按照特征上不同取值分好的数据字典
    :param feature: 特征在特征列表上的索引
    :param data_size: 总的数据集的大小
    :return: 条件熵
    """
    ent_a = 0
    feature_map = class_data_by_feature(datasets, feature)
    for x in list(feature_map.keys()):
        d_i = len(feature_map[x])
        ent_a += d_i / data_size * cal_empirical_entropy(class_data_by_class(feature_map[x]), d_i)
    return ent_a


def find_best_feature(datasets, class_map, data_size, feature_list):
    """
    找寻在当前数据集合上，用来分类的最佳特征
    :param datasets: 当前待分类的数据集
    :param class_map: 按照类别分好的数据字典
    :param data_size: 数据集合大小
    :param feature_list: 还可以用来分类的特征在原本特征列表上的索引
    :return: 最佳特征索引
    """
    g_max = -1  # 最大的信息增益
    result = -1  # 最佳特征的索引
    for x in range(len(feature_list)):
        # 计算信息增益
        g = cal_empirical_entropy(class_map, data_size) - cal_conditional_entropy(datasets, feature_list[x], data_size)
        # print(g)
        if g > g_max:
            g_max = g
            result = x
    return feature_list[result]


def create_tree(tree_node, datasets, feature_list):
    """
    创建决策树
    :param tree_node: 当前需要处理的节点，一开始调用时将根节点传入
    :param datasets: 需要处理的数据集合
    :param feature_list: 所有可用特征在特征列表上的索引
    """
    class_map = class_data_by_class(datasets)  # 对数据集按照类别进行分类
    if len(class_map.keys()) == 1:  # 如果当前数据集只有一个类别，表示已经是叶子节点了
        tree_node.node_class = "end"
        tree_node.name = list(class_map.keys())[0]
    else:
        best_feature = find_best_feature(datasets, class_map, len(datasets), feature_list)  # 否则计算最佳特征的索引
        tree_node.node_class = "decision"
        tree_node.name = labels[best_feature]
        tree_node.feature = best_feature
        feature_list.remove(best_feature)  # 继续向下分裂生长时，可供选择的特征中，删除当前节点使用了的特征
        feature_map = class_data_by_feature(datasets, best_feature)  # 根据该最佳节点划分数据，继续进行生长
        lists = sorted(list(feature_map.keys()))  # 排序可供该特征可供选择的值，用以方便后续的预测
        for x in range(len(lists)):  # 生成相应的子节点
            tree_node.next.append(node())
            create_tree(tree_node.next[x], feature_map[lists[x]], feature_list.copy())  # 注意这里使用拷贝


def predict(decision_tree, sample):
    """
    根据决策树进行预测
    :param decision_tree: 已经生成的决策树
    :param sample: 需要预测的样本
    :return: 最终返回叶子节点的 name 属性，代表最终的分类结果
    """
    while decision_tree.node_class == "decision":  # 如果当前节点仍然是判断节点，则继续向下
        decision_tree = decision_tree.next[sample[decision_tree.feature]]  # decision_tree.feature代表该节点使用的特征的索引
    print(decision_tree.name)  # 最终输出分类结果


if __name__ == "__main__":
    dataset, labels = createDataSet()  # 获取数据集和标签集，用于训练决策树
    tree = node()  # 生成决策树根
    create_tree(tree, dataset, list(range(len(labels))))
    predict(tree, [1, 1, 2])  # 预测一位大于三十岁且高收入的女性是否会买车，输出为是
