import numpy as np
from sys import argv
from numpy import mean, sqrt, std
from sklearn.linear_model import Perceptron
from sklearn import metrics


# Load data set
datafilename = 'sampleData1.txt'
print('Loading', datafilename)
dataset = np.loadtxt(datafilename, delimiter=',')
(numSamples, numFeatures) = dataset.shape
data = dataset[:,range(2)].reshape((numSamples, 2))
labels = dataset[:, 2].reshape((numSamples,))
classif = Perceptron()

print('Fitting a', type(classif).__name__, 'model to the dataset')]

## We have to reshape the data here because partial_fit expects a 2D arrray
##for the X input and an array for the Y input

reshapedData = data.reshape(1000,1,2)
reshapedLabel = labels.reshape(1000,1)
y = reshapedLabel
y_index, x_index, callsToPf, totalLoops, errorRate = 0,0,0,0,1
while(errorRate != 0):
    errorRate = 0
    for x in reshapedData:
        classif.partial_fit(x,y[y_index],classes = [-1,1])
        callsToPf += 1
        preds = classif.predict(data)
        errorRate = metrics.zero_one_loss(y,preds)
        print(errorRate)
        y_index += 1
        if(errorRate == 0):
            break
    totalLoops += 1
    y_index = 0


print("It took ", callsToPf, " weight updates to get the errorRate to 0")
print("Final weight vector is ",classif.coef_)
# x1 and x2 are the weights
x1 = classif.coef_[0][0]
x2 = classif.coef_[0][1]
#coeff is w0
coeff = classif.intercept_
print("intercept is ",coeff)
print("The equation of the decision boundary, setting x1 = x and x2 = y is\n h(x) = ",coeff[0], " + ", x1,"x + ",x2, "y ")




