#! /bin/env python
# -*- coding: utf-8 -*-
# import math
# import numpy as np
from numpy import *


def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) # 1.0是常数项
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    """
    梯度上升,缺陷：每次更新回归系数都需要遍历整个数据集
    :param dataMatIn:
    :param classLabels:
    :return:
    """
    # 转换成Numpy矩阵数据类型
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1)) # 系数初始化为1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights += alpha*dataMatrix.transpose()*error
    return weights # 画图用法plotBestFit(weights)


def stocGradAscent(dataMatrix, classLabels):
    """
    随机梯度上升,一次仅用一个样本点来更新回归系数,但会错分较多样本
    :param dataMatrix:
    :param classLabels:
    :return:
    """
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights += alpha*error*dataMatrix[i]
    return weights # 画图用法plotBestFit(weights)


def stoGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    改进的随机梯度上升,alpha在每次迭代时都会调整,随机选取样本来更新回归系数
    由于常数项,alpha永远不会减小到0,是为了保证在多次迭代后新数据仍然具有一定影响力
    j,i共同作用,避免了参数的严格下降,避免参数的严格下降也常见于模拟退火优化算法
    随机选取参数来更新回归系数,将减少周期性的波动.
    :param dataMatrix:
    :param classLabels:
    :param numIter:
    :return:
    """
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01 # 并不单调递减
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights += alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def plotBestFit(wei):
    import matplotlib.pyplot as plt
    # weights = wei.getA() # mat才有这个属性，将mat转换成array
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i] == 1):
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s') # 散点图
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1) # x方向范围, 0.1步长
    y = (-wei[0]-wei[1]*x)/wei[2] # 最佳拟合直线
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


