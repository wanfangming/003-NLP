#!/usr/bin/env python
# coding: utf-8
import numpy as np
import bayes


def trainNB0(trainMatrix, trainCategory):
    """

    :param trainMatrix: 文件单词矩阵
    :param trainCategory: 文件对应的类别
    :return:
    """
    numTrainDocs = len(trainMatrix)  # 文件数
    numWords = len(trainMatrix[0])  # 总单词数,注：这里应该是词汇表中的总单词数，直接叫总单词书很容易误解，在此样例中为32

    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 侮辱性文件出现的概率

    p0Nums = np.zeros(numWords)  # numWords为32
    p1Nums = np.zeros(numWords)

    p0Denom = 0.0
    p1Denom = 0.0

    for i in range(numTrainDocs):  # 此处为6
        if trainCategory[i] == 1:
            p1Nums += trainMatrix[i]  # 累加辱骂词的频次
            p1Denom += sum(trainMatrix[i])  # 对每篇文章的辱骂的频次进行统计汇总
        else:
            p0Nums += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    p1Vect = p1Nums / p1Denom
    # 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    p0Vect = p0Nums / p0Denom
    print('p0Vect: {}'.format(p0Vect))
    print('p1Vect: {}'.format(p1Vect))
    print('pAbusive: {}'.format(pAbusive))
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    使用算法：
        # 将乘法转换为加法
        乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
        加法：P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    :param vec2Classify: 待测数据[0,1,1,1,1...]，即要分类的向量
    :param p0Vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    :param p1Vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    :param pClass1: 类别1，侮辱性文件的出现概率
    :return: 类别1 or 0
    """
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1) # P(w|c1) * P(c1) ，即贝叶斯准则的分子
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1) # P(w|c0) * P(c0) ，即贝叶斯准则的分子·
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    # 1. 加载数据集
    listOPosts, listClasses = bayes.loadDataSet()
    print('listOPosts: ', listOPosts, '\n************************************\nlistClasses: ', listClasses)

    # 2. 创建单词集合
    myVocabList = bayes.createVocabList(listOPosts)

    # 3. 计算单词是否出现并创建数据矩阵
    trainMat = []
    for postinDoc in listOPosts:
        # 返回m * len(myVocabList)的矩阵，记录的都是0，1信息
        # print('postinDoc:', postinDoc)
        trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))

    # 4. 训练数据

    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))

    # 5. 测试数据
    testEntry = ['love', 'my', 'dalmatioin']
    thisDoc = np.array(bayes.setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(bayes.setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

testingNB()
