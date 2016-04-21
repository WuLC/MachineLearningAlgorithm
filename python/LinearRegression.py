# -*- coding: utf-8 -*-
# @Author: WuLC
# @Date:   2016-04-03 13:03:24
# @Last modified by:   LC
# @Last Modified time: 2016-04-12 20:48:48
# @Email: liangchaowu5@gmail.com
# @Function: linear regression with gradient descent of two methods: batch gradient descent and stochastic gradient descent 
# @Referer: http://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy


from sklearn.linear_model import SGDClassifier
import numpy as np
import random

def batchGradientDescent(x,y,theta,alpha):
    """batch gradient descent for linear regression
    
    @parms x: input of independent variables
    @parms y: input dependent variables
    @parms theta: weights parameterizing the hypothesis
    @parms alpha: learning rate
    """
    m,n = np.shape(x)
    xTran = x.transpose()
    convergence = 0.000000001
    lastCost = 0
    cost = -1    
    recurseCount = 0
    while abs(lastCost - cost) > convergence: # rcurse until converge
        lastCost = cost
        hypothesis = np.dot(x,theta)
        loss = hypothesis - y
        cost = np.sum(loss**2)/(2*m)
        gradient = np.dot(xTran,loss)/m
        theta = theta - alpha*gradient
        recurseCount += 1
    return recurseCount,theta
    
def stochasticGradientDescent(x,y,theta,alpha):
    """stochastic gradient descent for linear regression
    
    @parms x: input of independent variables
    @parms y: input dependent variables
    @parms theta: weights parameterizing the hypothesis
    @parms alpha: learning rate
    """
    m,n = np.shape(x)
    convergence = 0.000000001
    lastCost = 0
    cost = -1    
    recurseCount = 0
    while abs(lastCost - cost) > convergence: # rcurse until converge
        lastCost = cost
        hypothesis = np.dot(x,theta)        
        for i in range(m):
            # alpha = 4.0 / (1.0 +  i) + 0.01 
            loss = hypothesis[i] - y[i]
            # gradient = np.dot(x[i],loss)
            gradient = x[i,:].transpose() * loss            
            theta = theta - alpha * gradient
        cost = np.sum((hypothesis-y)**2)/(2*m)
        recurseCount += 1
    return recurseCount,theta
            
def getData(m,bias,variance):
    """
    get sample data for the test
    
    @params m:number of input example
    @params n:number of independent variables
    """
    x = np.zeros(shape=(m,2))
    y = np.zeros(m)
    for i in range(m):
        x[i][0] = 1
        x[i][1] = i
        y[i] = i^2+i
    return x,y


if __name__ == '__main__':
    x, y = getData(100,25,10)
    m, n = np.shape(x)
    alpha = 0.0005
    theta = np.ones(n)
    # recurseNum, theta = batchGradientDescent(x, y, theta, alpha)
    recurseNum, theta = stochasticGradientDescent(x,y,theta,alpha)    
    print recurseNum,theta

    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    # SGDClassifier throws error when there are float numbers in the training data set
    Y = np.array([0.1, 1.1, 2.1, 2]) 
    
    clf = SGDClassifier(loss ="log")
    clf.fit(x,y)
    # compare manually implemented SGD with SGDClassifier in sklearn
    for i in X:
        print i
        print 'manual method:',np.dot(theta,i)
        print 'sklearn method:', clf.predict(i)