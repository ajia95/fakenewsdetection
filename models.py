import numpy as np
import linearRegression as linReg
import logisticRegression as logReg
import data
import os
from sklearn.metrics import f1_score


path = os.path.abspath("")



def train(model, x, y, xv, yv):
	#linear regression --> 0
	#logistic regression --> 1
	coffs = []
	if model == 0:
		coffs = linReg.train(x,y,xv,yv)
	else:
		coffs = logReg.train(x,y, xv, yv)

	return coffs

def test(model, x, y, coffs):

	correct = 0
	size = len(y)
	preds = []
	for i in range(0, size):
		#linear regression --> 0
		#logistic regression --> 1
		prediction = linReg.classify(x[i,:], coffs) if model == 0 else logReg.classify(x[i,:], coffs)
		preds.append(prediction)
		#print ("prediction: ", prediction)
        
		if prediction == y[i]:
			correct = correct + 1
            
	#print ("coffs: ", coffs)
	if model == 0:
		print ("Results for Linear Regression: ")
	else:
		print ("Results for logistic regression: ")

	print ("accuracy = ", correct/size)
	print ("f1 scores: ", f1_score(y, preds, average=None))


def addOnes(x):
	r = np.size(x,0)
	new_x = np.concatenate((np.ones((r,1)), x),axis=1)
	return new_x

#get data
x,y = data.calcfeatures(path+'/data/training/training_stances.csv', path+'/data/training/train_bodies.csv')
x = addOnes(x)
#discard some features
x = x[:,[0,1,5,7]]



xv,yv = data.calcfeatures(path+'/data/training/validation_stances.csv', path+'/data/training/train_bodies.csv')
xv = addOnes(xv)
#discard some features
xv = xv[:,[0,1,5,7]]



#get test data
xt, yt = data.calcfeatures(path+'/data/testing/competition_test_stances.csv', path+'/data/testing/competition_test_bodies.csv')
xt = addOnes(xt)
#discard some features
xt = xt[:,[0,1,5,7]]



#train linear regression model
linModel = train(0,x,y,xv,yv)

#test model
test(0,xt,yt,linModel)

#train logistic regression model
logModel = train(1,x,y,xv,yv)

#test model
test(1,xt,yt,logModel)



