"""
Naive Bayes 朴素贝叶斯
Machine Learning 作业
使用朴素贝叶斯分类器实现简单的文本分类
@Author 黄伟
"""
from typing import Dict


class bayes:
    def __init__(self):
        self.train_dataset = [
            [["Chinese", "Beijing", "Chinese"], "C"],
            [["Chinese", "Chinese", "Shanghai"], "C"],
            [["Chinese", "Macao"], "C"],
            [["Tokyo", "Japan", "Chinese"], "J"]
        ]
        self.test_dataset = [
            ["Chinese", "Chinese", "Chinese", "Tokyo", "Japan"],
            ["Tokyo", "Tokyo", "Japan", "Shanghai"]
        ]
        self.total_doc = len(self.train_dataset)
        self.class_vector = []
        self.unique_word = []
        self.__process_class()
        self.frequency = {}  # type: Dict[str,Dict]
        self.total_frequency = {}  # type: Dict[str,int]
        self.__process_frequency()
        self.__process_total_frequency()
        self.laplace_x = 1
        self.laplace_y = len(self.unique_word)
        self.multinomial_probability = {}  # type: Dict[str,Dict]
        self.__change_to_multinomial_probability()
        print(self.multinomial_probability)
        self.__multinomial_probability_predict()

    def __process_total_frequency(self):
        for class_name in self.class_vector:
            count = 0
            for item in self.frequency[class_name].keys():
                if item != "doc":
                    count += self.frequency[class_name][item]
                    if item not in self.unique_word:
                        self.unique_word.append(item)
            self.total_frequency[class_name] = count

    def __change_to_multinomial_probability(self):
        for key in self.class_vector:
            self.multinomial_probability[key] = {}
            self.multinomial_probability[key]["doc"] = self.frequency[key]["doc"] / self.total_doc
            for item in self.unique_word:
                if item in self.frequency[key].keys():
                    self.multinomial_probability[key][item] = (self.frequency[key][item] + self.laplace_x) / \
                                                              (self.total_frequency[key] + self.laplace_y)
                else:
                    self.multinomial_probability[key][item] = self.laplace_x / \
                                                              (self.total_frequency[key] + self.laplace_y)

    def __process_frequency(self):
        """
        处理每个分类的频率，和属于各个分类的词频
        """
        for class_name in self.class_vector:
            self.frequency[class_name] = {}
            for x in self.train_dataset:
                if x[1] == class_name:
                    if "doc" not in self.frequency[class_name].keys():
                        self.frequency[class_name]["doc"] = 1
                    else:
                        self.frequency[class_name]["doc"] += 1
                    for item in x[0]:
                        if item in self.frequency[class_name].keys():
                            self.frequency[class_name][item] += 1
                        else:
                            self.frequency[class_name][item] = 1

    def __process_class(self):
        """
        获取训练数据集的类型列表
        :return:
        """
        for x in self.train_dataset:
            if x[1] not in self.class_vector:
                self.class_vector.append(x[1])

    def __multinomial_probability_predict(self):
        for text in self.test_dataset:
            for class_name in self.class_vector:
                probability = 1
                for x in text:
                    probability *= self.multinomial_probability[class_name][x]
                probability *= self.multinomial_probability[class_name]['doc']
                print(text, "属于 ", class_name, "的概率大小为: ", probability)


b = bayes()
# print(b.class_vector)
# print(b.frequency)
# print(b.total_frequency)
# print(b.multinomial_probability)
