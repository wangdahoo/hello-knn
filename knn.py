# coding=utf8

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import math

# 解决中文乱码
myfont = fm.FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

# 创建数据集
def create_datasets():
    # 分别为吃冰激凌数、喝水数、户外活动小时数
    datasets = np.array([
        [8, 4, 2],
        [7, 1, 1],
        [1, 4, 4],
        [3, 0, 5]
    ])

    labels = ['非常热', '非常热', '一般热', '非常热']

    return datasets, labels

# 可视化分析数据
def analyze_data_plot(x, y):
    fig = plt.figure()
    # 将画布划分为1行1列1块
    ax = fig.add_subplot(111) 
    ax.scatter(x, y)
    plt.title(u'游客冷热感知散点图', fontsize=18, fontproperties=myfont)

    # 设置标题和坐标
    plt.xlabel(u'吃冰激凌数', fontproperties=myfont)
    plt.ylabel(u'喝水数', fontproperties=myfont)

    # 保存
    plt.savefig(r'/Users/tom/github/hello-knn/datasets_plot.png', bbox_inches='tight')
    plt.show()

'''KNN 分类器'''
def knn_classifier(newV, datasets, labels, k):
    # 1 计算新的数据与样本数据之间的距离
    dist = euclidean_distance2(newV, datasets)
    # 2 根据距离进行排序
    sortedDist = dist.argsort(axis = 0)
    # 3 针对k个点，统计各个类别的数量
    classCount = {}
    for i in range(k):
        # sortedDist[i] 即样本的索引
        votelabel = labels[sortedDist[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    # 4 投票机制，根据少数服从多数原则确定分类
    sortedCount = sorted(classCount.items(), reverse=True)
    print('results:\n', dist, sortedDist, classCount, sortedCount[0][0])

    return sortedCount[0][0]

'''欧式距离计算'''
def euclidean_distance(instance1, instance2, length):
    d = 0
    for x in range(length):
        d += math.pow(instance1[x] - instance2[x], 2)
    return math.sqrt(d)

def euclidean_distance2(newV, datasets):
    rowsize, colsize = datasets.shape
    diff = tile(newV, (rowsize, 1)) - datasets
    # 对向量差值平方求和再开方
    diffMat = diff ** 2
    # axis: 0按列 1按行 
    return diffMat.sum(axis = 1) ** 0.5


if __name__ == '__main__':
    datasets, labels = create_datasets()

    print(datasets, '\n')
    print(labels, '\n')

    analyze_data_plot(datasets[:,0], datasets[:,1])

    d = euclidean_distance([2,4,4], [7,1,1], 3)
    print(d)

    d2 = euclidean_distance2([2,4,4], datasets)
    print(d2)

    c = knn_classifier([2,4,4], datasets, labels, 3)
    print(c)
