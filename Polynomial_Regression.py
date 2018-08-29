# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 09:44:21 2018

@author: acakmak
"""

#We start by creating some data set that we want to build
#a polynomial regression model:
#First import the libraries
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

np.random.seed(2)

pageSpeeds = np.random.normal(3.0, 1.0, 100)
purchaseAmount = np.random.normal(50.0, 30.0, 100) / pageSpeeds


scatter(pageSpeeds, purchaseAmount)


#Now we'll split the data in two - 80% of it will be used
#for "training" our model, and the other 20%
#for testing it. This way we can avoid overfitting.
trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]

trainY = purchaseAmount[:80]
testY = purchaseAmount[80:]

#And then our trainig dataset looks like as below:
scatter(trainX, trainY)

#And our test dataset looks like as below:
scatter(testX, testY)


#Now we'll try to fit an 7th-degree polynomial to this data
x = np.array(trainX)
y = np.array(trainY)

p4 = np.poly1d(np.polyfit(x, y, 7))

#And plot our training dataset
xp = np.linspace(0, 7, 100)
axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0, 200])
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='b')
plt.show()


#And plot our testing dataset
testx = np.array(testX)
testy = np.array(testY)

axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0, 200])
plt.scatter(testx, testy)
plt.plot(xp, p4(xp), c='b')
plt.show()


#It looks like a good model for our dataset but when we print tthe r^2_score
#we can see it is not great model.
from sklearn.metrics import r2_score

r2 = r2_score(testy, p4(testx))

print(r2)


#The training dataset has slightly better r^2_score:
from sklearn.metrics import r2_score

r2 = r2_score(np.array(trainY), p4(np.array(trainX)))

print(r2)



