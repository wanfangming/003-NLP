#!/usr/bin/env python
# coding: utf-8

from nltk.corpus import brown, names
import nltk
import random

# 从names库中获取姓名及性别，共7944条数据
names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])


def gender_features(word):
    return {'last_letter': word[-1]}


random.shuffle(names)   # 将数据打乱

featuresets = [(gender_features(n), g) for (n, g) in names]  # 做数据集，格式为list中存放7994条元祖

train_set, test_set = featuresets[500:], featuresets[:500]  # 训练数据取前500后的7500条左右，测试数据去前五百

classifier = nltk.NaiveBayesClassifier.train(train_set)

print(classifier.classify(gender_features('Neo')))
print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(5)