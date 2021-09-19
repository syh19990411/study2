import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = r'C:\Users\Ted\Desktop\data\数据集.txt'#用显示声明字符串不用转义（）加r
data = pd.read_csv(path, header=None, names=['Population','Profit'])
data.head()
data.plot(kind='scatter', x='Population', y='Profit', figsize=(8, 4))
def computeCost(X,y,theta):
    inner = np.power(((X * theta.T)-y),2)
    return np.sum(inner) / (2 * len(X))
#这个部分是计算J（Ѳ)，X是矩阵
data.insert(0,'Ones',1)
cols = data.shape[1]
X = data.iloc[:,:-1]#除最后一列保存
y = data.iloc[:,cols-1:cols]#仅要最后一列
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))#需要学习的参数
computeCost(X, y, theta)
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])#计算参数的数量
    cost = np.zeros(iters)#储存误差函数
    for i in range(iters):
        error = (X * theta.T) - y#计算误差
        for j in range(parameters):
            term = np.multiply(error, X[:, j])#取x所有行的第j列
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * sum(term))
        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost
alpha = 0.01
iters = 1500
g, cost = gradientDescent(X, y, theta, alpha, iters)
print('g的值：'  , g)#θ的值
predict1 = [1,3.5]*g.T
print("predict1:",predict1)
predict2 = [1,7]*g.T
print("predict2:",predict2)
#画图
x = np.linspace(data.Population.min(), data.Population.max(),100)
f = g[0,0] + (g[0, 1] * x)#θ0和θ1x
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(x, f, 'r',label='Prediction')
ax.scatter(data.Population,data.Profit,label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
#plt.show()
#多变量线性回归
path = r"C:\Users\Ted\Desktop\data\数据集2.txt"
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedroom', 'Price'])
print(data2.head())
#特征归一化
data2 = (data2 - data2.mean()) / data2.std()
print(data2.head())
#梯度下降

#加一列常数列
data2.insert(0, 'Ones', 1)

#初始化x和y
clos = data2.shape[1]
X2 = data2.iloc[:,0:clos-1]
y2 = data2.iloc[:,cols-1:cols]


#转换成matrix格式，初始化theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

#运行梯度算法
g2,cost2 = gradientDescent(X2, y2, theta2, alpha, iters)
print(g2)
#正规方程
def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y#@X.T@X等价与X.T.dot(x)
    return theta
final_theta1 = normalEqn(X,y)#这里用的是data1的数据
print(final_theta1)
final_theta2 = normalEqn(X2,y2)#这里用的是data2的数据
print(final_theta2)


