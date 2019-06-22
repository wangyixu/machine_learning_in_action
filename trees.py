# -*- coding:utf-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import numpy as np


def getNumLeafs(myTree):
    numLeafs = 0  # 初始化叶子
    firstStr = next(iter(
        myTree))  # python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]  # 获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0  # 初始化决策树深度
    firstStr = next(iter(
        myTree))  # python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]  # 获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth  # 更新层数
    return maxDepth


"""
函数说明:绘制结点

Parameters:
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 结点格式
Returns:
    无

"""


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")  # 定义箭头格式
    font = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc", size=14)  # 设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',  # 绘制结点
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


"""
函数说明:标注有向边属性值

Parameters:
    cntrPt、parentPt - 用于计算标注位置
    txtString - 标注的内容
Returns:
    无

"""


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算标注位置
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


"""
函数说明:绘制决策树

Parameters:
    myTree - 决策树(字典)
    parentPt - 标注的内容
    nodeTxt - 结点名
Returns:
    无

"""


def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")  # 设置叶结点格式
    numLeafs = getNumLeafs(myTree)  # 获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)  # 获取决策树层数
    firstStr = next(iter(myTree))  # 下个字典
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制结点
    secondDict = myTree[firstStr]  # 下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key], cntrPt, str(key))  # 不是叶结点，递归调用继续绘制
        else:  # 如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


"""
函数说明:创建绘制面板

Parameters:
    inTree - 决策树(字典)
Returns:
    无

"""


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')  # 创建fig
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;  # x偏移
    plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
    plt.show()  # 显示绘制结果


"""
函数说明:计算给定数据集的经验熵(香农熵)

Parameters:
    dataSet - 数据集
Returns:
    shannonEnt - 经验熵(香农熵)

"""


def calcShannonEnt(dataSet):
    '''计算香农熵
    '''
    labelCounts = {}  # 保存每个标签(Label)出现次数的字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0  # 香农熵
    numEntires = len(dataSet)  # 数据集的行数
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSetForSeries(dataSet, axis, midVal):
    '''将数据集划分为给定特征 <= 和 > midVal 的两个子集
    '''
    leDataSet, gtDataSet = [], []
    for featVec in dataSet:
        if featVec[axis] <= midVal:
            leDataSet.append(featVec)
        else:
            gtDataSet.append(featVec)
    return leDataSet, gtDataSet


def chooseBestFeatureAndMidValToSplit(dataSet):
    '''分别计算各个特征各中间值划分下的信息增益，返回最佳特征index和mid值
    '''
    numFeatures = len(dataSet[0]) - 1  # 特征数量，最后一列是class
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain, bestFeature, bestMid = -1000.0, -1, -1

    for featIndex in range(numFeatures):
        featList = [example[featIndex] for example in dataSet]  # 当前特征值列表
        classList = [example[-1] for example in dataSet]  # 分类列表
        dictList = dict(zip(featList, classList))
        sortedFeatList = sorted(dictList.items(), key=operator.itemgetter(0))  # 按照连续值的大小排列
        numOfFeatList = len(sortedFeatList)
        midFeatList = [round((sortedFeatList[i][0] + sortedFeatList[i + 1][0]) / 2.0, 3) for i in
                       range(numOfFeatList - 1)]  # 计算划分点，保留三位小数
        # 计算出各个划分点的信息增益
        for mid in midFeatList:
            leDataSet, gtDataSet = splitDataSetForSeries(dataSet, featIndex, mid)  # 将连续值划分为不大于当前划分点和大于当前划分点两部分

            # 计算两部分的特征值熵和权重的乘积之和
            newEntropy = len(leDataSet) / len(sortedFeatList) * calcShannonEnt(leDataSet) + len(gtDataSet) / len(
                sortedFeatList) * calcShannonEnt(gtDataSet)
            infoGain = baseEntropy - newEntropy  # 计算出信息增益
            # print('特征' + str(i) + '当前划分值为：' + str(mid) + '，此时的信息增益为：' + str(infoGain))
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = featIndex
                bestMid = mid
    print('最大信息增益为以 特征' + str(bestFeature) + '、划分值为' + str(bestMid) + ' 划分：' + str(bestInfoGain))
    return bestFeature, bestMid


def majorityCnt(classList):
    '''统计classList中出现此处最多的元素(类标签)
    '''
    classCount = {}
    for vote in classList:  # 统计classList中每个元素出现的次数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 根据字典的值降序排序
    return sortedClassCount[0][0]  # 返回classList中出现次数最多的元素


def createSeriesTree(dataSet, labels):
    '''生成决策树
    '''
    classList = [example[-1] for example in dataSet]  # 标签列表
    print('剩余数据集行数：' + str(len(classList)))

    if classList.count(classList[0]) == len(classList):
        print('只有一个类别，属于：' + str(classList[0]))
        return classList[0]

    bestFeat, bestMid = chooseBestFeatureAndMidValToSplit(dataSet)

    if bestFeat == -1:
        major = majorityCnt(classList)
        print('不可分了，大部分属于：' + str(major))
        return major

    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    # del (labels[bestFeat])     连续值特征不删除

    leDataSet, gtDataSet = splitDataSetForSeries(dataSet, bestFeat, bestMid)
    myTree[bestFeatLabel]['<=' + str(bestMid)] = createSeriesTree(leDataSet, labels)
    myTree[bestFeatLabel]['>' + str(bestMid)] = createSeriesTree(gtDataSet, labels)
    return myTree




import re
import random

def classify(inputTree, labels, testVec):
    # print(labels)
    firstStr = next(iter(inputTree))  # 获取决策树结点
    # print(firstStr)
    secondDict = inputTree[firstStr]  # 下一个字典
    # print(secondDict)
    featIndex = labels.index(firstStr)  # 索引此标签对应的index
    # print(featIndex)
    midVal = float(re.sub("<=", "", list(secondDict.keys())[0]))
    # print(midVal)
    if testVec[featIndex] <= midVal:
        key = '<=' + str(midVal)
    else:
        key = '>' + str(midVal)

    if type(secondDict[key]).__name__ == 'dict':  # value的类型还是字典说明还没结束，结束时类型是 数字
        classLabel = classify(secondDict[key], labels, testVec)
    else:
        classLabel = secondDict[key]
    return classLabel


def file2matrix(filename):
    '''读取文件数据，返回特征矩阵和标签向量
    '''
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # 获取文件行数
    returnMat = [[] for _ in range(numberOfLines)]

    fr = open(filename)
    index = 0
    for line in fr.readlines():
        odom = line.strip().split()  # 将单个数据分隔开存好
        nums_float = map(float, odom)  # 转化为浮点数
        for f in nums_float:
            returnMat[index].append(f)
        index += 1
    return returnMat




if __name__ == '__main__':
    filename = "/Users/yixu/Downloads/第2次作业/Page Blocks Classification Data Set/page-blocks.data"
    dataSet = file2matrix(filename)  # 打开并处理数据
    random.shuffle(dataSet)
    print(len(dataSet))
    labels = ['HEIGHT', 'LENGTH', 'AREA', 'ECCEN', 'P_BLACK',
              'P_AND', 'MEAN_TR', 'BLACKPIX', 'BLACKAND', 'WB_TRANS']
    featLabels = []

    myTree = createSeriesTree(dataSet[400:], labels)
    print(myTree)
    #createPlot(myTree)

    for num in [100, 200, 300, 400]:
        errorCount = 0.0
        for i in range(num):
            testVec = dataSet[i]
            result = classify(myTree, labels, testVec[:10])
            if result != testVec[-1]:
                errorCount += 1
            # print("分类结果:%s\t  真实类别:%s" % (result, testVec[-1]))
        print("测试个数：%3d   错误个数：%2d   准确率:%f%%" % (num, errorCount, (num - errorCount) / float(num) * 100))