# 导入包含有 KNN 算法的模块 neighbors
from sklearn import neighbors
# 导入数据集 datasets 模块
from sklearn import datasets


# 调用 KNN 分类器
knn = neighbors.KNeighborsClassifier()


# datasets.load_iris() 返回数据库
iris = datasets.load_iris()


print(iris)


# KNN 建模，iris.data：150*4（萼片长度、萼片宽度、花瓣长度、花瓣宽度） 的特征向量；iris.target：150*1 的特征向量：每一行的花都是什么类别
knn.fit(iris.data, iris.target)
# 根据已有模型预测新的对象
predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])
# [0] 代表 setosa 类
print(predictedLabel)
