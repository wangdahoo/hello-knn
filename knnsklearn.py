# coding=utf8

from numpy import *
import numpy as np
from sklearn import neighbors

def create_datasets():
		# 分别为吃冰激凌数、喝水数、户外活动小时数
    datasets = np.array([
        [8, 4, 2],
        [7, 1, 1],
        [1, 4, 4],
        [3, 0, 5],
        [8, 5, 2]
    ])

    labels = ['非常热', '非常热', '一般热', '非常热', '非常热']

    return datasets, labels

def knn_predict(newV):
	knn = neighbors.KNeighborsClassifier()
	datasets, labels = create_datasets()
	knn.fit(datasets, labels)
	predictRes = knn.predict([newV])
	return predictRes

if __name__ == '__main__':
	print(knn_predict([2, 4, 4]))