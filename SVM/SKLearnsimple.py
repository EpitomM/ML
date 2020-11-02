from sklearn import svm

# 定义三个点，x1(2,0)、x2(1,1)、x3(2,3)
x = [[2, 0], [1, 1], [2, 3]]
# y 为三个实例所对应的分类标记，此例子中只有两个类（绿色直线左边、绿色直线右边两类），分别使用 0、1 代表
y = [0, 0, 1]
# 通过 SVM 建立模型。kernel='linear' 线性盒函数
clf = svm.SVC(kernel='linear')
clf.fit(x, y)

print(clf)

# get support vectors 支持向量
print(clf.support_vectors_)
# get indices of support vectors 支持向量在数组x中的下标
print(clf.support_)
# get number of support vectors for each class 每个类中支持向量的个数（一共两个类，每个类有一个支持向量）
print(clf.n_support_)
# (2,0) 这个点属于哪一类
print(clf.predict([[2, 0]]))