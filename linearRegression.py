import numpy as np
import math
import sys

def test(x, y, coffs):

	correct = 0
	size = len(y)
	for i in range(0, size):

		prediction = classify(x[i,:], coffs)
		
		if prediction == y[i]:
			correct = correct + 1
	#print ("ac: ", correct/size)
	return correct/size

def hypothesis(x, coffs):
	#f = no. of coefficients (no.of features + 1) 
	#f = len(x)
	h = np.dot(x,coffs)
	#print ("hypo: ", h) 
	return h

def cost(x,y,coffs):
	#row for each sample
	#column for each feature
	#m is number of samples
	m = np.size(y)
	#m = len(y)
	cost = 0
	for i in range(0,m):
		cost = cost + (hypothesis(x[i,:], coffs)-y[i])**2
	cost = cost/(2*m)
	return cost


def diff_cost(x,y,coffs, j):

	#m is number of samples
	m = np.size(y)
	#m = len(y)
	cost = 0
	for i in range(0,m):
		cost = cost + (hypothesis(x[i,:], coffs)-y[i])*x[i,j]
	cost = cost/m
	return cost


def batchGradientDescent(a, f, x, y, epochs,xv,yv,batchSize):
	train_accuracy = 0
	validation_accuracy = 0
	coffs = np.random.rand(f)

	batches = math.ceil(len(y)/batchSize)
    
	#while true:
	for it in range(0,epochs):
		#print ("gradient descending...")
		#until converence
		#if cost(x,y,coffs)<=target:
		#	break
		#minimise this
		#cost = cost(x,y,coffs)

		for b in range(0,batches):
            
			temp = np.zeros(f)
			start = b*batchSize
			end = start + batchSize 
			for j in range(0,f):
			   #print ("f: ", j)                
			   diffcost = diff_cost(x[start:end,:],y[start:end],coffs,j)
			   #print ("diff cost: ", diffcost)            
			   temp[j] = coffs[j] - a*diffcost
	
			coffs = temp[:]
			#print ("coffs: ", coffs) 
        
        
        
        
    
		tAc = test(x,y,coffs)
		vAc = test(xv,yv,coffs)
		
		if tAc>train_accuracy and vAc<=validation_accuracy:
			#print ("sttttttooooooooppppp")      	     
			break
		else:
			train_accuracy = tAc
			validation_accuracy = vAc

	return coffs


def classify(x,coffs):
	h = hypothesis(x, coffs)
	return min([0,1,2,3], key=lambda x:abs(x-h))

def train(x,y,xv,yv):
	a = 0.005
	epochs = 30
	batchSize = 50  
	f = np.size(x,1)
	coffs = batchGradientDescent(a,f,x,y,epochs,xv,yv,batchSize) 
	return coffs

