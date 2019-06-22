# coding:utf-8
__author__ = 'Yixu Wang'

import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt


'''
函数说明：加载数据文件，返回i属性字典和数据矩阵

Parameters:
    fileName - 文件名
Returns:
    attribute_dict - 返回的属性 序号-字符串 字典
    data_mat - 返回的特征矩阵

'''
def load_data(filename):
    fr = open(filename)
    attribute_dict = {}
    data_mat = []
    line = fr.readline()
    while line != '':
        vals = line.strip().split(',')
        if vals[0] == "A":
            attribute_dict[int(vals[1])] = vals[3] + ',' + vals[4]
            # print(vals[1], vals[3], vals[4])
            line = fr.readline()
        elif vals[0] == "C":
            arr = [0.0] * 298
            line = fr.readline()
            while line.startswith("V"):
                vals = line.strip().split(',')
                arr[int(vals[1]) - 1000] = 1.0  # 属性序号 1000-1297 -> 0-297
                line = fr.readline()
            data_mat.append(arr)
        else:
            line = fr.readline()
    return attribute_dict, data_mat


'''
函数说明：选取合适的 k 值，保留 ration 以上的能量比例

Parameters:
    sigma - 奇异值列表
    ratio - 需要保留的能量比例
Returns:
    k - 保留的奇异值个数

'''
def selectK(sigma, ratio=0.9):
    sorted_sigma = sorted(sigma, reverse=True)
    print('奇异值从大到小排序列表：\n', sorted_sigma)

    Sig2 = [x ** 2 for x in sorted_sigma]
    print('奇异值的平方：\n', Sig2, '\n')
    threshold = sum(Sig2) * ratio
    add_val = 0
    k = 0
    for i in range(len(Sig2)):
        add_val += Sig2[i]
        k += 1
        if add_val >= threshold:
            break
    print('如图计算可知，当保留比例为', ratio, '时，需要保留的奇异值个数为', k, '\n')
    return k


'''
函数说明：对输入数据集进行SVD分解，实现矩阵的降维和恢复。
    包括：对奇异值的平方作图分析，选取合适的 k 值，保留 ration 以上的能量比例。

Parameters:
    dataMat - 特征矩阵
    topNfeat - 保留的维度
Returns:
    lowDDataMat - 降维后的特征矩阵
    reconMat - 恢复维度后的特征矩阵

'''
def svd(dataMat, ratio=0.9):
    U, Sigma, VT = svds(np.mat(dataMat), 294)
    sorted_sigma = sorted(Sigma, reverse=True)
    print('奇异值从大到小排序列表：\n', sorted_sigma)

    Sig2 = [x ** 2 for x in sorted_sigma]
    print('奇异值的平方：\n', Sig2, '\n')
    threshold = sum(Sig2) * ratio
    add_val = 0
    k = 0
    for i in range(len(Sig2)):
        add_val += Sig2[i]
        k += 1
        if add_val >= threshold:
            break
    print('如图计算可知，当保留比例为', ratio, '时，需要保留的奇异值个数为', k, '\n')

    # 作图
    svIndex = [i for i in range(1, 295)]
    percent = [20, 40, 60, 100]
    # 生成图表
    plt.plot(svIndex, Sig2 / sum(Sig2) * 100)
    # 设置横坐标为year，纵坐标为population，标题为Population year correspondence
    plt.xlabel('Index')
    plt.ylabel("Percent of SV's Square")
    plt.title('SVD Diagram')
    # 设置纵坐标刻度
    plt.xticks([i for i in range(0, 295, 50)])
    plt.yticks([0, 5, 10, 15, 20, 25])
    # 设置填充选项：参数分别对应横坐标，纵坐标，纵坐标填充起始值，填充颜色（可以有更多选项）
    #plt.fill_between(eigValsIndex, sorted_eigVals, 10, color='green')
    # 显示图表
    plt.show()
    #print('如果保留 95% 以上的方差占比，即占特征值和95%的主成分特征值，则需保留', k, '个主成分特征\n')

    rec_sigma = np.eye(k, k)
    for i in range(k):
        rec_sigma[i][i] = sorted_sigma[i]
    # print(rec_sigma)

    # 恢复矩阵
    reconMat = np.dot(np.dot(U[:,:k], rec_sigma), VT[:k,:])
    reconMat[reconMat < 0.5] = 0
    reconMat[reconMat >= 0.5] = 1

    return reconMat


if __name__ == '__main__':
    file_name = 'anonymous-msweb.data'
    attribute_dict, dataMat = load_data(file_name)
    print("属性个数:", len((attribute_dict)), ",  数据集shape:", np.shape(dataMat), '\n')
    print('第一条原始数据的前20项：\n', dataMat[0][:20], '\n')

    U, Sigma, VT = svds(np.mat(dataMat), 294)
    print(Sigma)
    reconMat = svd(dataMat, 0.9)    # SVD算法对数据集进行降维和恢复
    print('原始数据矩阵与恢复矩阵各列差值平均值前20项：\n', (dataMat - reconMat).mean(0)[:20], '\n')