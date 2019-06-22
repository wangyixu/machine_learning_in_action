# -*- coding:utf-8 -*-
from math import log
import random


"""
函数说明:读取.data文件，实现数据到特征矩阵和标签向量的转换，包括数据顺序的随即打乱等

Parameters:
    filename - 文件名字
Returns:
    dataMat - 有效数据的特征矩阵
    labelVec - 有效数据的标签向量

"""
def file2matrix(filename):
    mat = []
    fr = open(filename)
    for line in fr.readlines():
        strs_in_line = line.strip().split(',')  # 将单个数据分隔开存好
        mat.append(strs_in_line)

    random.shuffle(mat) #随即打乱数组

    data_mat = []
    labels_vec = []
    for arr in mat:
        data_mat.append(arr[:8])
        labels_vec.append(arr[-1])

    return data_mat, labels_vec


"""
函数说明:将输入的数据集合转化为向量表示

Parameters:
    featuresMat - 特征的名称、类型矩阵
    inputSet - 输入数据集合
Returns:
    returnVec - 返回向量集合

"""
def setOfWords2Vec(featuresMat, inputSet):
    n = len(inputSet)
    returnVec = [0] * n
    if n == 8:
        for i in range(n):
            featureVal = inputSet[i]
            if featureVal in featuresMat[i]:
                returnVec[i] = featuresMat[i].index(featureVal)
            else:
                print("the word: %s is not in my Vocabulary!" % featureVal)
    else:
        for i in range(n):
            returnVec[i] = featuresMat.index(inputSet[i])
    return returnVec


"""
函数说明:计算概率 P_Ck, k=0,1,...,4, 类别为 k 的概率;
               P_ajl_in_Ck, 类别为 k 的情况下第 j 个特征为 l 的概率

Parameters:
    trainMat - 输入训练数据矩阵
    classVec - 输入数据标签向量
Returns:
    P_Ck - 类别为 k 的概率
    P_ajl_in_Ck - 类别为 k 的情况下第 j 个特征为 l 的概率

"""
def trainNB(trainMat, classVec):
    numOfData = len(trainMat)
    numOfFeatures = len(trainMat[0])

    classCnt = [1.0] * 5
    for i in range(5):
        classCnt[i] += float(classVec.count(i))
    P_Ck = [cnt / (numOfData + 5) for cnt in classCnt]

    # P_Ajl_in_Ck = [[np.ones(5)] * numOfFeatures] * 5
    P_Ajl_in_Ck = [[[1.0 for _ in range(5)] for _ in range(numOfFeatures)] for _ in range(5)]
    # print(P_Ajl_in_Ck[0])
    for i in range(numOfData):
        cate_k = classVec[i]
        for j in range(numOfFeatures):
            P_Ajl_in_Ck[cate_k][j][trainMat[i][j]] += 1

    for i in range(5):
        for j in range(numOfFeatures):
            # print('---')
            for k in range(5):
                P_Ajl_in_Ck[i][j][k] = log(P_Ajl_in_Ck[i][j][k] / (classCnt[i] + 3))
                # print(P_Ajl_in_Ck[i][j])

    return P_Ajl_in_Ck, P_Ck


"""
函数说明:使用bayes算法对输入的一个数据集合进行计算，并根据最大概率进行分类

Parameters:
    vec2Classify - 输入进行预测的数据集合
    P_Ck - 类别为 k 的概率
    P_ajl_in_Ck - 类别为 k 的情况下第 j 个特征为 l 的概率
Returns:
    class - 类别

"""
def classifyNB(vec2Classify, P_Ajl_in_Ck, P_Ck):
    p = [0.0] * 5
    for i in range(5):
        sum = log(P_Ck[i])
        for j in range(8):
            sum += P_Ajl_in_Ck[i][j][vec2Classify[j]]
        p[i] = sum
    # print(p)
    return p.index(max(p))


'''
函数说明:线性回归对数据进行分析和预测，采用“留出法”将数据集分割为训练集和测试集，
    采用多次训练和预测取平均值的方式获得平均准确率，且每次会随即打乱数据

Parameters:
    trial_times - 训练和预测的次数
    num_of_test - 测试集个数
    filename - 文件名
    featuresMat - 特征矩阵
    category - 类别向量
Returns:
    accuracy - 平均预测准确率

'''
def Naive_Bayes_Test(trial_times, num_of_test, filename, featuresMat, category):
    count = 0
    for time in range(trial_times):
        dataMat, classVec = file2matrix(filename)  # 打开并处理数据

        # 转换成向量矩阵、标签向量
        vecDataMat = []
        for arr in dataMat:
            vecDataMat.append(setOfWords2Vec(featuresMat, arr))
        classVec = setOfWords2Vec(category, classVec)

        P_Ajl_in_Ck, P_Ck = trainNB(vecDataMat[num_of_test:], classVec[num_of_test:])
        # print(theta)
        cnt = 0.0
        for i in range(num_of_test):
            a = classifyNB(vecDataMat[i], P_Ajl_in_Ck, P_Ck)
            if a == classVec[i]:
                cnt += 1
            # print('实际评分是 %f , 预测评分是 %f , 误差值是 %f' % (labelVec[i], predict, error))
        # print('平均误差值是 %f' % (cnt / 100))
        print('第 %2d 次测试误差小于1的个数  %d/%d' % (time+1, cnt, num_of_test))
        count += cnt
    return count / (trial_times * num_of_test)




if __name__ == '__main__':
    filename = "/Users/yixu/Downloads/Data Analysis/hw03/贝叶斯/数据/nursery.data"
    featuresMat = [['usual', 'pretentious', 'great_pret'],
                   ['proper', 'less_proper', 'improper', 'critical', 'very_crit'],
                   ['complete', 'completed', 'incomplete', 'foster'],
                   ['1', '2', '3', 'more'],
                   ['convenient', 'less_conv', 'critical'],
                   ['convenient', 'inconv'],
                   ['nonprob', 'slightly_prob', 'problematic'],
                   ['recommended', 'priority', 'not_recom']]
    category = ['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior']

    # 使用算法对数据集进行10次训练和预测，预测集大小为100
    accuracy = Naive_Bayes_Test(20, 100, filename, featuresMat, category)
    print('平均预测准确率 %f%%' % (accuracy * 100))