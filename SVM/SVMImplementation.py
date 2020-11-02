from __future__ import print_function

# 某些步骤需要计时，看他花费了多少时间
from time import time
# 打印程序进展信息
import logging
# 把识别出来的人脸进行绘制：看预测出来的人脸和实际人脸是否符合
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays
# 下载人脸数据集：对于每个人，有很多张人脸照片
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
# 得到特征向量矩阵：每一行是一个实例，每一列是一个特征值。
X = lfw_people.data
# 特征向量的维度（每张人脸图片的维度：对于每个人提取的特征值个数）
n_features = X.shape[1]

# the label to predict is the id of the person
# 标记值：每个实例对应的类别分类
y = lfw_people.target
# 类别里面有哪几个名字
target_names = lfw_people.target_names
# 总共有多少个类，即所有人脸照片对应的一共是多少人（比如100张照片，每人10张，共对应10人）
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
# 把数据集拆分为训练集和测试集：X_train 训练集特征向量；y_train 训练集类别标记值
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)


###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
# 记录初始时间（计算耗时）
t0 = time()
# X_train高维特征向量降为低维
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

# 提取特征点（提取人脸的特征值）
eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
# 高维矩阵转化为低维矩阵
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


###############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
# 看哪一对C和gamma的组合(这里共5*6=30种组合)能产生更好的精确度。gamma：特征值的比例
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
# 通过SVM进行建模：找到使得margin最大能划分两类点的超平面
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
# 预测新来的特征向量对应的是哪一类
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

# 分类器预测准确率。y_test测试集真实分类；y_pred预测分类；target_names真实标签中对应的姓名。
print(classification_report(y_test, y_pred, target_names=target_names))
# 建立n*n的方格，填充上预测的标记和真实的标记，看预测准确率，矩阵对角线上的数据对应的值就是预测准确的。
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
