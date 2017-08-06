#! /bin/env python
# -*- coding: utf-8 -*-

from math import log
import operator
import pickle


def calcShannonEnt(dataSet):
    """
    计算给定数据集的熵
    :param dataSet:
    :return:
    """
    # 为所有可能分类创造字典
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 1
        else:
            labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    """
    创建简单鱼鉴定数据表
    :return:
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    划分数据集
    :param dataSet:待划分的数据集
    :param axis:划分数据集的特征
    :param value:特征的返回值
    :return:
    """
    retDataSet = [] # 去除value后的list对象
    for featVec in dataSet:
        if featVec[axis] == value:
            # 抽取
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec) # 这里的extend和append的区别要搞清
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    :param dataSet:
    :return: 最合适的划分特征
    """
    numFeatures = len(dataSet[0]) - 1 # 去除label列
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) # 将重复的值删掉，只剩set([0,1])
        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
    当处理完所有属性，但类标签依旧不是唯一时，采用多数表决发
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    创造树
    :param dataSet:
    :param labels:
    :return:
    """
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止继续划分,list.count统计某个元素在列表中出现次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征后，仅剩1个标签项，返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优特征,标签列中删除最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 获取特征标签对应的值,递归调用
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,
                                                               bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    使用决策树的分类算法
    :param inputTree:
    :param featLabels:
    :param testVec:
    :return:
    """
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    # 将标签字符串转换成索引,index()返回索引位置，没有找到则抛出异常
    featIndex = featLabels.index(firstStr)
    classLabel = ""
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# 之前每次使用分类器doug需要重新构造决策树，这是非常耗时的
# 下面使用pickle序列化模块,在硬盘上存储决策树的分类器


def storeTree(inputTree, filename):
    """
    存储
    :param inputTree:
    :param filename:
    :return:
    """
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    """
    加载
    :param filename:
    :return:
    """
    fr = open(filename)
    return pickle.load(fr)





