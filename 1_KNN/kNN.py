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

