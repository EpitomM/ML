# 支持矩阵运算
import numpy as np
# 画图功能
import pylab as pl
from sklearn import svm

# we create 40 separable points
# 训练实例集合。randn(20, 2)：20个2维的点。- [2, 2]：负号表示产生的随机点靠下方，均值是2，方差是2
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
# 把前20个点归类为0，后20个点归类为1
Y = [0]*20 +[1]*20

#fit the model 建立SVM模型
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# get the separating hyperplane
# 由 w_0x + w_1y +w_3=0 可推导出 y = -(w_0/w_1) x + (w_3/w_1)
w = clf.coef_[0]
# 要画出的区分两类的直线的斜率
a = -w[0]/w[1]
# 从(-5,5)范围内产生一些连续的xx值
xx = np.linspace(-5, 5)
# 把xx带入，画出yy斜线
yy = a*xx - (clf.intercept_[0])/w[1]

# plot the parallels to the separating hyperplane that pass through the support vectors
# 画出与支持向量相切的两条线，这三条线平行：斜率相同、截距不同
# 获取模型中的第一个支持向量，把它的值带入方程，算出下面的线的截距
b = clf.support_vectors_[0]
yy_down = a*xx + (b[1] - a*b[0])
# 获取模型中的另外一个支持向量，把它对应的值带入方程，计算出上面的线的截距
b = clf.support_vectors_[-1]
yy_up = a*xx + (b[1] - a*b[0])

print("w: ", w)
print("a: ", a)

# print "xx: ", xx
# print "yy: ", yy
print("support_vectors_: ", clf.support_vectors_)
print("clf.coef_: ", clf.coef_)

# switching to the generic n-dimensional parameterization of the hyperplan to the 2D-specific equation
# of a line y=a.x +b: the generic w_0x + w_1y +w_3=0 can be rewritten y = -(w_0/w_1) x + (w_3/w_1)


# plot the line, the points, and the nearest vectors to the plane
# 边际最大化的线
pl.plot(xx, yy, 'k-')
# 与支持向量相切的两条线
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
          s=80, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()