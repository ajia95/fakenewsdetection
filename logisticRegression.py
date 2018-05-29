import numpy as np 
#import models
import math

def test(x, y, coffs,cat):
	correct = 0
	size = len(y)
	for i in range(0, size):
		prediction = cat if hypothesis(x[i,:], coffs)>0.5 else -1
		
		if prediction == y[i]:
			correct = correct + 1
	return correct/size

def sigmoid(z):
  return 1/ (1 + np.exp(-z))


def cost(x,y, coffs,cat):
	#row for each sample
	#column for each feature
	#m is number of samples
	m = np.size(y)
	#m = len(y)
	cost = 0
	for i in range(0,m):
		c = (-np.log(hypothesis(x[i,:],coffs))) if y == cat else (-np.log(1-hypothesis(x[i,:],coffs)))
		cost = cost + c
	cost = cost/m
	return cost
	

def hypothesis(x, coffs):
	return sigmoid(np.dot(x,coffs))


def diff_cost(x,y,coffs, j,cat):
	#row for each sample
	#column for each feature
	#m is number of samples
	m = np.size(y)
	#m = len(y)
	cost = 0
	for i in range(0,m):
		y_ = 1 if y[i] == cat else 0
		cost = cost + (hypothesis(x[i,:], coffs)-y_)*x[i,j]
	cost = cost/m
	return cost


def batchGradientDescent(a, f, x, y, epochs,cat,xv,yv,batchSize):
	#print("gradient descending...")
	train_accuracy = 0
	validation_accuracy = 0
	coffs = np.random.rand(f)
	batches = math.ceil(len(y)/batchSize)
	#while true:
	for it in range(0,epochs):
		#until converence
		#if cost(x,y,coffs,cat)<=target:
		#	break
		#minimise this
		#cost = cost(x,y,coffs)
		for b in range(0,batches):
			start = b*batchSize
			end = start + batchSize
			temp = np.zeros(f)
			for j in range(0,f):
			   temp[j] = coffs[j] - a*diff_cost(x[start:end,:],y[start:end],coffs,j,cat)
			coffs = temp[:]
        
        
		tAc = test(x,y,coffs,cat)
		vAc = test(xv,yv,coffs,cat)

		if tAc>train_accuracy and vAc<=validation_accuracy:
			break
		else:
			train_accuracy = tAc
			validation_accuracy = vAc
		
		
	return coffs


def classify(x, coffss):
	h = []
	for c in range(0,4):
		h.append(hypothesis(x, coffss[c]))
	#print (h)    
	return h.index(max(h))


def train(x,y, xv, yv):
	a = 0.005
	epochs = 30
	batchSize = 50    
	f = np.size(x,1)
	coffss = []
	cats = 4
	for c in range(0,cats):
		coffss.append(batchGradientDescent(a,f,x,y,epochs,c,xv,yv,batchSize))
	return coffss
