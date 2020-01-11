import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp


def sigmoidFunction(X):
    denominator = 1.0 + np.e ** (-1.0 * X)
    g = 1.0 / denominator
    return g


def featureMapping(a1, a2):
    a1.shape = (a1.size, 1)
    a2.shape = (a2.size, 1)
    vector = np.ones(shape=(a1[:, 0].size, 1))
    for i in range(1, 7):
        for j in range(i + 1):
            q = (a1 ** (i - j)) * (a2 ** j)
            vector = np.append(vector, q, axis=1)
    return vector


fin = np.loadtxt('ex2data2.txt', dtype=float, delimiter=',')
matrix = np.loadtxt('ex2data2.txt', delimiter=',')
results = pd.read_csv('ex2data2.txt', sep=',', header=None)
X = matrix[:, 0:2]
Y = matrix[:, 2]
m, n = X.shape
lmbda = 1
jError = 0.0
thetha_X = 0.0
PosX = []
PosY = []
NegX = []
NegY = []
learningRate = 6
maxIterations = 5000
errorRate = 0.000000001
convergeCheck = False
iterationCount = 0
featurizedVector = featureMapping(X[:, 0], X[:, 1])
r, c = featurizedVector.shape
Y.shape = (m, 1)
thethaAtStart = np.zeros(shape=(featurizedVector.shape[1], 1))
thethaOnUpdate = np.zeros(shape=(featurizedVector.shape[1], 1))

# Calculating initial cost
for j in range(r):
    for i in range(c):
        thetha_X = thetha_X + thethaAtStart[i] * featurizedVector[j][i]
    hx = 1.0 * (thetha_X)
    gz = sigmoidFunction(hx)
    thetha_X = 0.0
    param1 = (-Y[j]) * math.log(gz)
    param2 = 1.0 * (1 - Y[j]) * math.log(1 - gz)
    error = param1 - param2
    jError = jError + error
    thethaSum = 0.0
for i in range(c):
    if (c != 0):
        thethaSum = 1.0 * (thethaSum + 1.0 * (thethaAtStart[i] ** 2))
J = (1.0 / (r)) * jError
newFactor = 1.0 * lmbda / (2 * r) * 1.0
mul = newFactor * thethaSum
initialCost = mul + J
print("Initial cost comes out as: ", initialCost)

# Computing gradient descent for regularized logistic regression
m = r
jError = 0.0
thetha_X = 0.0
finalResult = 0.0
while not convergeCheck:
    for j in range(c):
        # updating all theetas except 0th one
        if (j != 0):
            for w in range(r):
                for i in range(c):
                    thetha_X = thetha_X + thethaAtStart[i] * featurizedVector[w][i]
                hx = 1.0 * (thetha_X)
                thetha_X = 0.0
                gz = sigmoidFunction(hx)
                mulResult = 1.0 * (gz - Y[w]) * (featurizedVector[w][j] * 1.0)
                finalResult = finalResult + mulResult
        regularizedExp = 1.0 * (1.0 / r) * finalResult
        finalResult = 0.0
        updatedVal = regularizedExp + 1.0 * (lmbda * 1.0 / r) * thethaAtStart[j]
        thethaOnUpdate[j] = updatedVal
        # updating 0th theeta
        if (j == 0):
            for k in range(r):
                for i in range(c):
                    thetha_X = featurizedVector[k][i] * thethaAtStart[i] + thetha_X
                gz = sigmoidFunction(1.0 * (thetha_X))
                thetha_X = 0.0
                mulResult = 1.0 * (gz - Y[k]) * (featurizedVector[k][0] * 1.0)
                finalResult = finalResult + mulResult;
        regularizedExp = (1.0 / r) * finalResult
        finalResult = 0.0
        updatedVal = regularizedExp
        thethaOnUpdate[0] = updatedVal
    for features in range(c):
        thethaAtStart[features] = thethaAtStart[features] - 1.0 * (learningRate * thethaOnUpdate[features])

    # calculating cost again for comparison purposes
    jError = 0.0
    thetha_X = 0.0
    for j in range(r):
        for i in range(c):
            thetha_X = thetha_X + thethaAtStart[i] * featurizedVector[j][i]
        gz = sigmoidFunction(1.0 * (thetha_X))
        thetha_X = 0.0
        param1 = (-Y[j]) * math.log(gz)
        param2 = 1.0 * (1 - Y[j]) * math.log(1 - gz)
        error = param1 - param2
        jError = jError + error
    thethaSum = 0.0
    for i in range(c):
        if (c != 0):
            thethaSum = 1.0 * (thethaSum + 1.0 * (thethaAtStart[i] ** 2))
    J = jError * 1.0 / (r)
    jError = 0.0
    mul = 1.0 * lmbda / (2 * r) * 1.0 * thethaSum
    comparisonCost = mul + J
    print(comparisonCost)
    if abs(initialCost - comparisonCost) <= errorRate:
        print(thethaAtStart)
        print('Converged at iteration number: ', iterationCount)
        convergeCheck = True
    initialCost = comparisonCost
    iterationCount = iterationCount + 1
    if iterationCount == maxIterations:
        convergeCheck = True
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-0.8, 1.2, 50)
z = np.zeros(shape=(len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i, j] = (featureMapping(np.array(u[i]), np.array(v[j])).dot(np.array(thethaAtStart)))
z = z.T
mp.contour(u, v, z)
r, c = fin.shape
labelledMatrix = fin[:, c - 1]
x1 = results[0]
x2 = results[1]
MeanX1 = np.mean(x1, axis=0)
StdDevX1 = np.std(x1, axis=0)
MeanX2 = np.mean(x2, axis=0)
StdDevX2 = np.std(x2, axis=0)
for i in range(len(labelledMatrix)):
    if (labelledMatrix[i] == 1):
        PosX.append(x1[i]);
        PosY.append(x2[i])
for i in range(len(labelledMatrix)):
    if (labelledMatrix[i] == 0):
        NegX.append(x1[i]);
        NegY.append(x2[i])
mp.title('Decision Boundary for Lambda = 1 ', fontsize=12)
mp.xlabel('Microchip Test 1')
mp.ylabel('Microchip Test 2')
mp.plot(PosX, PosY, 'k+', label="Y=1")
mp.plot(NegX, NegY, 'yo', label="Y=0")
mp.legend(loc=4)
mp.show()