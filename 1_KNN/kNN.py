#! /bin/env python
# -*- coding: utf-8 -*-

from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    :param inX: 用于分类的输入向量
    :param dataSet: 输入的训练样本集
    :param labels: 标签向量
    :param k: 选择最近邻居的数目
    :return:
    """
    # 计算欧式距离
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet # 复制数据Size行1列
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) # axis=0--多个列表中对应位置求和,=1,单个列表中所有元素求和
    distances = sqDistances**0.5
    # 选择距离最小的k个点
    sortedDistIndicies = distances.argsort() # 有小到大给出索引值,可以是float
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1 # dict的get函数，返回键的值，如果无则返回默认值0
    # 排序
    # sorted重新生成一个list,reverse=True降序,key是第一个元素
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    """
    :param filename: 文件名的字符串
    :return: 训练样本矩阵和类标签向量
    """
    fr = open(filename)
    arrayLines = fr.readlines()
    numberLines = len(arrayLines)
    returnMat = zeros((numberLines, 3)) # 创建返回的NumPy矩阵
    classLabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip() # 去除开头和结尾处的空格
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3] # 每行前3列
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autonorm(dataSet):
    """
    归一化特征值
    :param dataSet:
    :return:
    """
    minVals = dataSet.min(0) # 参数0从列中选取最小值而不是行
    maxVals = dataSet.max(0) # minVal maxVal都是1行3列
    ranges = maxVals - minVals
    # normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1)) # 特征值相除
    return normDataSet, ranges, minVals

def datingClassTest():
    """
    数据集10%作为测试集,90%作为训练集
    :return:
    """
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autonorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0 # 计数分错的个数
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is %f" % (errorCount/float(numTestVecs))

def classifyPerson():
    """
    输入一个人的参数,判断此人喜欢程度
    :return:
    """
    resultList = ["一点也不喜欢", "略微喜欢", "非常喜欢"]
    percentTats = float(raw_input("玩电子游戏时间比："))
    ffMiles = float(raw_input("每年获得的飞行公里数："))
    iceCream = float(raw_input("每年吃的冰淇淋数："))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autonorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print "你对这个人的喜欢程度为：", resultList[classifierResult - 1]

