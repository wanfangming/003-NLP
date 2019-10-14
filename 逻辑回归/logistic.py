import numpy as np
import matplotlib.pyplot as plt
import time


def load_data(file_name='logisticTestSet.txt'):
    """
    收集并准备数据，共一百条，返回类型为list
    :param file_name: 训练文件名
    :return:
    """
    data_mat = []
    label_mat = []
    file = open(file_name)
    for line in file.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[-1]))
    return data_mat, label_mat


def sigmoid(inX):
    # return 1.0 / (1 + exp(-inX)
    # Tanh是Sigmoid的变形，与 sigmoid 不同的是，tanh 是0均值的。因此，实际应用中，tanh 会比 sigmoid 更好。
    return 1.0 / (1 + np.exp(-inX))


def grad_ascent(data_mat_in, class_labels):
    """
    梯度上升法的实现
    :param data_mat_in: NumPy数组，每行代表每个训练样本，每列代表一项特征
    :param class_labels: 类别标签
    :return:
    """
    data_mat = np.mat(data_mat_in)  # 将数据转换为numpy矩阵
    label_mat = np.mat(class_labels).transpose()

    m, n = np.shape(data_mat)  # m为样本数，n为特征数

    alpha = 0.001  # 移动步长，即就是学习率

    # 迭代次数
    max_cycles = 1000

    weights = np.ones((n, 1))

    for k in range(max_cycles):
        h = sigmoid(data_mat * weights)

        error = label_mat - h  # 向量相减，label_mat是实际值
        weights = weights + alpha * data_mat.transpose() * error  # 计算偏移量，得出回归系数
    return np.array(weights)


# 随机梯度下降法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    # n*1的矩阵
    # 函数ones创建一个全1的数组
    weights = np.ones(n)   # 初始化长度为n的数组，元素全部为 1
    for i in range(m):
        # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn,此处求出的 h 是一个具体的数值，而不是一个矩阵
        h = sigmoid(sum(dataMatrix[i]*weights))
        # print 'dataMatrix[i]===', dataMatrix[i]
        # 计算真实类别与预测类别之间的差值，然后按照该差值调整回归系数
        error = classLabels[i] - h
        # 0.01*(1*1)*(1*n)
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def plot_best_fit(data_arr, label_mat, weights):
    """
    将得到的结果展示出来
    :param data_arr: 样本数据的特征
    :param label_mat: 样本数据的类别标签，即目标变量
    :param weights: 回归系数
    :return: None
    """
    n = len(data_arr[0])
    data_arr = np.array(data_arr)

    # 这是干嘛的？这个函数画的不知所云诶
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(100):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# 使用逻辑回归进行分类
def test_lr():
    # 1.收集并准备数据，共一百条，返回类型为list
    data_mat, label_mat = load_data()
    print('data_mat的类型：', type(data_mat))

    # 2.训练模型
    data_arr = np.array(data_mat)  # 将list类型转换为array类型，其中的值是未变的
    print('data_arr的类型：', type(data_arr))
    weights = grad_ascent(np.array(data_arr), label_mat)

    print('weight为：', weights)  # 怎么都是1...？

    # 数据可视化
    plot_best_fit(data_mat, label_mat, weights)
    arr = (np.dot(data_mat, weights))
    print(arr)


now = time.time()  # 计算时间开销用
test_lr()
time = time.time() - now
print('\n所花时间:{}'.format(time))
