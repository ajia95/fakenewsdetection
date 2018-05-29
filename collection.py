import csv
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import nltk
import os
nltk.download('stopwords')
nltk.download('punkt')

path = os.path.abspath("")

stop_words = set(stopwords.words('english'))


def getDocCount(bodiesFile):
    with open(bodiesFile, 'r', encoding='UTF-8') as csvDataFile: 
         
        csvReader = csv.reader(csvDataFile)
        row_count = sum(1 for row in csvReader)
    print (row_count-1)
    return row_count-1

def loadBodiesTokens(bodiesFile):
    tknzr = TweetTokenizer()
    bodies={}
    with open(bodiesFile, 'r', encoding='UTF-8') as csvDataFile: 
        csvReader = csv.reader(csvDataFile)
        first = 1
        for row in csvReader:
            if first==1:
                first = 0
            else:
                bodies[row[0]] = [token.lower() for token in tknzr.tokenize(row[1]) if (token.isalpha() and token not in stop_words)]
                           
    return bodies

def loadBodies(bodiesFile):
    bodies={}
    with open(bodiesFile, 'r', encoding='UTF-8') as csvDataFile: 
        csvReader = csv.reader(csvDataFile)
        first = 1
        for row in csvReader:
            if first==1:
                first = 0
            else:
                bodies[row[0]] = row[1]
                           
    return bodies



def getVocab():
	freq = []
	vocab = []
	length = 0
	tknzr = TweetTokenizer()
	with open(path+'/data/training/training_stances.csv', 'r', encoding='UTF-8') as csvDataFile: 
		csvReader = csv.reader(csvDataFile)
		first = 1
		for row in csvReader:
			if first == 1:
				first = 0
			else:
				headline = row[0]
				tokens = tknzr.tokenize(headline)
				tokens=[token.lower() for token in tokens if (token.isalpha() and token not in stop_words)]
				#for word in r.split(headline):
				length = length + len(tokens)
				for word in tokens:
					if word not in vocab:
						vocab.append(word)
						freq.append(1)
					else:
						ind = vocab.index(word)
						freq[ind] = freq[ind] + 1
				
	with open(path+'/data/training/train_bodies.csv', 'r', encoding='UTF-8') as csvDataFile: 
		csvReader = csv.reader(csvDataFile)
		first = 1
		for row in csvReader:
			if first == 1:
				first = 0
			else:
				body = row[1]
				tokens = tknzr.tokenize(body)
				tokens=[token.lower() for token in tokens if (token.isalpha() and token not in stop_words)]
				length = length + len(tokens)
				#for word in r.split(headline):
				for word in tokens:
					if word not in vocab:
						vocab.append(word)
						freq.append(1)
					else:
						ind = vocab.index(word)
						freq[ind] = freq[ind] + 1
	return vocab, freq, length


				
#vocab list
#vocab, freq, length = getVocab()