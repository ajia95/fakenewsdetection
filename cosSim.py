import csv
from string import punctuation
import re
import math
import numpy as np
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
nltk.download('punkt')
import os
from collection import loadBodies



path = os.path.abspath("")

r = re.compile(r'[\s{}]+'.format(re.escape(punctuation)))
stop_words = set(stopwords.words('english'))

def cosSim(vec1,vec2):
	innerP = np.inner(vec1,vec2)
	x = math.sqrt(np.sum(np.square(vec1)))
	y = math.sqrt(np.sum(np.square(vec2)))
	return innerP/(x+y)


def calcVecSims():
	tknzr = TweetTokenizer()  
	bigram_vectorizer = CountVectorizer(tokenizer=tknzr.tokenize, ngram_range=(1, 2), binary=False, lowercase=True, 
		stop_words='english', min_df=1)

	with open(path+'/data/training/training_stances.csv', 'r', encoding='UTF-8') as csvDataFile1: 
		 
			csvReader1 = csv.reader(csvDataFile1)
			bodies = loadBodies(path+'/data/training/train_bodies.csv')
			first = 1
			for row in csvReader1:

				if first == 1:
					first = 0
				else:
					corpus = []
					corpus.append(row[0])
					#headline = row1[0]
					#tokens = word_tokenize(headline)
					#tokens=[token.lower() for token in tokens if token.isalpha()]
					corpus.append(bodies[row[1]])		
							
					print ("cosine similarity:")
					vecs = bigram_vectorizer.fit_transform(corpus).toarray()
					#print (vecs)
					print (cosSim(vecs[0,:],vecs[1,:]))
				
				

calcVecSims()

