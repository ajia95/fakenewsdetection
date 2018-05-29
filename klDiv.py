import csv
#from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import math
import collection
import os

stop_words = set(stopwords.words('english'))
path = os.path.abspath("")


def dirichletSmooth(word, tok, vocab,freq,length):
	mu = 1
	lambdaa = len(tok)/(len(tok)+mu)

	#return lambdaa*(tok.count(word)/len(tok)) + (1-lambdaa)* len(tok) * (vocab.freq[collection.vocab.index(word)]/collection.length)
	return lambdaa*(tok.count(word)/len(tok)) + (1-lambdaa) * (freq[vocab.index(word)]/length)

def klDivergence(tokens1, tokens2, vocab,freq,length):
	ans = 0

	for v in vocab:
		
		#ans = ans + model(v, tokens1, len(tokens1))*math.log(model(v, tokens2, len(tokens2)))
		#ans = ans + lindstoneCorrection(v, tok1)*math.log(lindstoneCorrection(v, tok2))
		#ans = ans + dirichletSmooth(v, tokens1)* math.log(dirichletSmooth(v, tokens2))
		ans = ans + dirichletSmooth(v, tokens1, vocab,freq,length)* math.log10(dirichletSmooth(v, tokens2, vocab,freq,length))
	return -ans



def printKLDivs():
	bodies = collection.loadBodiesTokens(path+'/data/training/train_bodies.csv')
	vocab, freq, length = collection.getVocab()
    
	tknzr = TweetTokenizer()
	with open(path+'/data/training/training_stances.csv', 'r', encoding='UTF-8') as csvDataFile1: 
		csvReader1 = csv.reader(csvDataFile1)
		first = 1
		for row in csvReader1:

			if first == 1:
				first = 0
			else:
				tokens1 = tknzr.tokenize(row[0])
				tokens1=[token.lower() for token in tokens1 if (token.isalpha() and token not in stop_words)]

				tokens2 = bodies[row[1]]
							

				print (klDivergence(tokens1, tokens2,vocab,freq,length))			
											

printKLDivs()





