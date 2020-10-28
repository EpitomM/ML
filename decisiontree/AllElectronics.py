# sklearn 对数据输入格式有要求，只支持 Integer，不支持 youth、no 等非数值，所以需要使用 DictVectorizer 进行转化
from sklearn.feature_extraction import DictVectorizer
# 数据存在 csv 文件中，所以需要 import csv
import csv
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

# Read in the csv file and put features into list of dict and list of class label
# 读取csv文件并将功能放入字典列表和类标签列表
allElectronicsData = open(r'AllElectronics.csv', 'r')
# 按行读取
reader = csv.reader(allElectronicsData)
# csv 文件的第一行题头：['RID', 'age', 'income', 'student', 'credit_rating', 'class_buys_computer']
headers = next(reader)

print(headers)

# 预处理
# 将 age 转为三维变量 (youth middle_aged senior)。
# 比如 age = youth，转变后的三维向量为 (1 0 0)；如果 age = middle_aged，对应的三维向量为 (0 1 0)
# 题头 age,income,student,credit_rating,class_buys_computer 可以转为 youth middle_aged senior high medium low yes no fair excellent 多维向量
# 第一行数据 youth,high,no,fair,no 对应转换为 (1 0 0 1 0 0 0 1 1 0)
# 存放特征值：age、income、student、credit_rating
featureList = []
# 存放是否买电脑结果值
labelList = []

# 把原始值转化为包含字典的 list：[{'age': 'youth', 'income': 'high', ...}, {'age': 'youth', 'income': 'high',...}, ...]，然后调用 Python 自带方法 DictVectorizer() 将 list 转为多维向量形式
# 遍历每一行：
for row in reader:
    # 取每一行的最后一个值（是否买电脑），把它存放到 labelList 中
    labelList.append(row[len(row)-1])
    # 存放特征值：age、income、student、credit_rating
    rowDict = {}
    # i 从第一列 age 开始，到倒数第二列 credit_rating 为止
    for i in range(1, len(row)-1):
        # 例如：rowDict[age] = youth
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)

# Vetorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList) .toarray()

print("dummyX: " + str(dummyX))
print(vec.get_feature_names())

print("labelList: " + str(labelList))

# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " + str(dummyY))

# Using decision tree for classification 创建决策树
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))


# Visualize model
with open("allElectronicInformationGainOri.dot", 'w') as f:
    # feature_names=vec.get_feature_names()：将多维向量(1 0 0 1 0 0 0 1 1 0)转变成以前的特征值 age、income、student、credit_incoming
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

oneRowX = dummyX[0, :].reshape(1, -1)
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0][0] = 1
newRowX[0][2] = 0
print("newRowX: " + str(newRowX))

predictedY = clf.predict(newRowX)
print("predictedY: " + str(predictedY))


