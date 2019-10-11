#!/usr/bin/env python
# coding: utf-8


# 定义数据
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# In[9]:


def createVocabList(dataSet):
    """
    获取所有单词的集合
    :param dataSet: 所有文档的数据
    :return: list类型的所有文档的单词已去重集合
    """
    vocabSet = set([])
    for document in dataSet:
        # 操作符|用于求两个集合的并集
        vocabSet = vocabSet | set(document)
    print('\n所有单词的集合: ', vocabSet, '\n')
    return list(vocabSet)


# In[10]:


def setOfWords2Vec(vocabList, inputSet):
    """

    :param vocabList: 词汇列表
    :param inputSet: 每个单独的文章，形式为列表
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # print(vocabList.index(word))
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

