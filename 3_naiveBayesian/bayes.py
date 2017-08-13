#! /bin/env python
# -*- coding: utf-8 -*-
"""
朴素贝叶斯，朴素是指两个假设：
假设1：特征之间相互独立,样本数可以从N^1000减少到1000×N
假设2：每个特征同等重要
虽然上面两个假设都是有问题的，但是朴素贝叶斯的实际效果却很好
"""
from numpy import *
import random
import re
import feedparser


def loadDataSet():
    """
    创建一些实验样本
    :return:
    """
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    #1 代表侮辱性文字, 0 代表正常言论,6个文档是否是侮辱性言论,stupid
    return postingList, classVec


def createVocabList(dataSet):
    """
    创建一个包含在所有文档中出现的不重复词的列表
    :param dataSet:
    :return:
    """
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 创建两个集合的并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    将文档转换成词向量,词集模型,每个词只能出现一次
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return: 文档向量
    """
    # 创建一个其中所有元素都是0的向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    """
    朴素贝叶斯词袋模型,每个单词可以出现多次
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return: 文档向量
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
    :param trainMatrix: 文档矩阵
    :param trainCategory: 每篇文章类别标签构成的向量
    :return:
    """
    numTrainDocs = len(trainMatrix) # 文档数量
    numWords = len(trainMatrix[0]) # 每个文档有多少单词
    pAbusive = sum(trainCategory)/float(numTrainDocs) # 文档属于侮辱性的概率
    # 初始化概率
    # p0Num = zeros(numWords); p1Num = zeros(numWords)
    # p0Denom = 0.0; p1Denom = 0.0
    # 为避免多个概率相乘,其中一个概率为0,最后乘积也为0的问题
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        # 如果是侮辱性文字
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] # 向量相加
            p1Denom += sum(trainMatrix[i])
        # 如果是正常文字
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 对每个元素做除法
    #p1Vect = p1Num/p1Denom # 侮辱性文档概率，之后需要修改成log
    #p0Vect = p0Num/p0Denom # 正常文字概率，之后需要修改成log
    # 取对数可以避免下溢出或者浮点数舍入导致的错误,且加对数函数后走向和原函数相同,求最大值一致
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    :param vec2Classify: 文档的表示分量[0,0,1,1,0,1...]这种类型
    :param p0Vec: 训练好的p0向量
    :param p1Vec: 训练好的p1向量
    :param pClass1: 训练集中侮辱性文档比例
    :return:
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listPost, listClass = loadDataSet()
    myList = createVocabList(listPost)
    trainMat = []
    for postLine in listPost:
        trainMat.append(setOfWords2Vec(myList, postLine))
    p0V, p1V, PAb = trainNB0(trainMat, listClass)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, PAb)
    testEntry2 = ['stupid', 'garbage']
    thisDoc2 = array(setOfWords2Vec(myList, testEntry2))
    print testEntry2, 'classified as: ', classifyNB(thisDoc2, p0V, p1V, PAb)


"""
使用朴素贝叶斯对电子邮件进行分类
构建一个完整的程序对一组文档进行分类,将错分的文档输出到屏幕上
"""


# 测试函数,使用朴素贝叶斯进行交叉验证
def textParse(bigString):
    """
    文件解析函数
    :param bigString:
    :return:
    """
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2 ]


def spamTest():
    """
    完整的垃圾邮件测试函数
    :return:
    """
    docList = []; classList = []; fullText = []
    # 导入并解析文本文件
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50); testSet = []
    # 随机构建训练集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat = []; trainClass = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print 'classification error ' + str(docList[docIndex])
    print 'the error rate is : ', float(errorCount)/len(testSet)


"""
使用来自不同城市的广告训练一个分类器,然后观察分类器的效果.
我们的目的并不是使用该分类器进行分类,而是通过观察单词和条件概率
来发现与特定城市相关的内容.

使用算法:构建一个完整的程序,封装所有内容.给定两个RSS源,该程序会显示最常用的公共词
"""


# RSS源分类器及高频词去除函数
def calcMostFreq(vocabList, fullText):
    """
    计算最高频率
    :param vocabList:
    :param fullText:
    :return:
    """
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1),
                        reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    """
    两个RSS源
    :param feed1:
    :param feed0:
    :return:
    """
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    # print "length of feed1 " + str(len(feed1['entries'])) # 25
    # print "length of feed0 " + str(len(feed0['entries'])) # 25
    # 每次访问一条RSS源
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    # print top30Words # 列表中是二元元组
    # 去掉出现次数最高的那些词
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount)/len(testSet)
    return vocabList, p0V, p1V

"""
分析数据:显示地域相关的用词

先对向量pSF与pNY进行排序,然后按照顺序将词打印出来
"""


def getTopWord(ny, sf):
    """
    最具表征性的词汇显示函数
    :param ny:
    :param sf:
    :return:
    """
    import operator
    vocabLIst, p0V, p1V = localWords(ny, sf)
    topNY = []; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabLIst[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabLIst[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key= lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]


if __name__ == "__main__":
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    getTopWord(ny, sf)
