import numpy as np
import matplotlib.pyplot as plt

'''
函数说明：加载数据文件，并进行分割、处理，其中缺失值以 nan 替代，返回矩阵

Parameters:
    fileName - 文件名
    delim - 分割符
Returns:
    datArr - 返回的特征矩阵

'''
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.replace('?', 'nan').strip().split(delim) for line in fr.readlines()]
    #print(stringArr)
    #datArr = [list(map(float,line)) for line in stringArr]
    datArr = []
    for arr in stringArr:
        datArr.append(features2vec(arr))
    return np.mat(datArr)


'''
函数说明：对特征向量前7个特征和类别标签进行转换，转换为相应index值

Parameters:
    feature_list - 特征向量
Returns:
    vec_list - 转换后的特征向量

'''
def features2vec(feature_list):
    features = [[],
                [],
                ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
                'isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury',
                'mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche',
                'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo'],
                ['diesel', 'gas'],
                ['std', 'turbo'],
                ['four', 'two'],
                ['hardtop', 'wagon', 'sedan', 'hatchback', 'convertible'],
                ['4wd', 'fwd', 'rwd'],
                ['front', 'rear'],
                [],
                [],
                [],
                [],
                [],
                ['dohc', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'rotor'],
                ['eight', 'five', 'four', 'six', 'three', 'twelve', 'two'],
                [],
                ['1bbl', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi'],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                []]
    vec_list = []
    for i in range(len(features)):
        if len(features[i]) > 0 and feature_list[i] != 'nan':
            vec_list.append(features[i].index(feature_list[i]))
        else:
            vec_list.append(float(feature_list[i]))
    #print(vec_list)
    return vec_list


'''
函数说明：对特征向量前7个特征和类别标签进行转换，转换为相应index值

Parameters:
    feature_list - 特征向量
Returns:
    vec_list - 转换后的特征向量

'''
def replaceNanWithMean(fileName):
    datMat = loadDataSet(fileName, ',')
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[np.nonzero(np.isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat


'''
函数说明：
    PCA算法处理
    第一步：用样本数据减去样本均值；
    第二步：计算数据的主成分，由其协方差矩阵的特征向量按照对应的特征值大小排序得到。第一种方法是计算数据协方差矩阵。因为协方差矩阵是方阵，所以我们可以用前面的方法计算特征值和特征向量。
    第三步：建一个转换矩阵，它的每一列都是主成分的特征向量，第一主成分是最大特征值对应的特征向量。我们用特征向量中的前k个主成分作为转换矩阵，然后用数据矩阵右乘转换矩阵。

Parameters:
    dataMat - 特征矩阵
    topNfeat - 保留的维度
Returns:
    lowDDataMat - 降维后的特征矩阵
    reconMat - 恢复维度后的特征矩阵

'''
def pca(dataMat, topNfeat=9999999):
    meanVals = dataMat.mean(0)
    meanRemoved = dataMat - meanVals    # meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)  # 求协方差矩阵
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))    # 求特征值、特征向量
    eigValInd = np.argsort(eigVals)            # 特征值从小到大排列

    # 保留方差的百分比可以用选择的特征值的和与所有特征值的和来表示
    sorted_eigVals = sorted(eigVals, reverse=True)
    print('特征值列表：\n', sorted_eigVals, '\n')
    sum = np.sum(sorted_eigVals)
    print('各特征值代表方差占比列表：\n', sorted_eigVals / sum, '\n')

    # 保留 95%以上 的方差
    vals = 0
    k = 0
    for i in range(len(sorted_eigVals)):
        vals += sorted_eigVals[i]
        k += 1
        #print(k, ' ', vals, ' ', vals/sum)
        if vals >= 0.95 * sum:
            break

    eigValsIndex = [i for i in range(1, 27)]
    percent = [20, 40, 60, 100]
    # 生成图表
    plt.plot(eigValsIndex, sorted_eigVals / sum * 100)
    # 设置横坐标为year，纵坐标为population，标题为Population year correspondence
    plt.xlabel('Principal Component Number')
    plt.ylabel('Percentage of Variance')
    plt.title('PCA')
    # 设置纵坐标刻度
    plt.xticks([i for i in range(27)])
    plt.yticks([0, 20, 40, 60, 80, 100])
    # 设置填充选项：参数分别对应横坐标，纵坐标，纵坐标填充起始值，填充颜色（可以有更多选项）
    #plt.fill_between(eigValsIndex, sorted_eigVals, 10, color='green')
    # 显示图表
    plt.show()
    print('如果保留 95% 以上的方差占比，即占特征值和95%的主成分特征值，则需保留', k, '个主成分特征\n')

    eigValInd = eigValInd[:-(k+1):-1]  # 倒着取特征值序号，cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       # 根据序号，reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects     # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


if __name__ == '__main__':
    fileName = 'imports-85.data'
    dataMat = replaceNanWithMean(fileName)
    print('数据集特征矩阵的形状是： ', dataMat.shape, '\n')
    #print(dataMat.mean(0))

    lowDDataMat, reconMat = pca(dataMat)  # PCA算法对数据集进行降维

    print('降维后的前5条数据：\n', lowDDataMat[:5], '\n')
    print('原始数据矩阵与恢复矩阵各列差值平均值：\n', (dataMat - reconMat).mean(0), '\n')































