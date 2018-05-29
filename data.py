import os
import numpy as np
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import os
import csv
from collection import loadBodies
from collection import loadBodiesTokens
from collection import getDocCount
import features as feat
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

stop_words = set(stopwords.words('english'))


def calcfeatures(stancesFile, bodiesFile):
    path = os.path.abspath("")
    #gensim.models.KeyedVectors.load_word2vec_format
    #wmd_model = Word2Vec.load_word2vec_format('/data/w2v_googlenews/GoogleNews-vectors-negative300.bin.gz', binary=True)
    wmd_model = KeyedVectors.load_word2vec_format(path+'/data/GoogleNews-vectors-negative300.bin', binary=True)
    wmd_model.init_sims(replace=True)
    tknzr = TweetTokenizer()

    count = 0
    features = []
    classes = []

    #N = getDocCount(path+'/data/training/train_bodies.csv')

    keys = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}

    bodies = loadBodies(bodiesFile)

    bigram_vectorizer = CountVectorizer(tokenizer=tknzr.tokenize, ngram_range=(1, 2), binary=False, lowercase=True, 
        stop_words='english', min_df=1)
    
    vectorizer = TfidfVectorizer(tokenizer=tknzr.tokenize, ngram_range=(1, 1), binary=False, lowercase=True, 
        stop_words='english', min_df=1)

    tfidfMat = vectorizer.fit_transform(list(bodies.values()))
    tfidfMat = vectorizer.transform(list(bodies.values()))
    tfidfMat = tfidfMat.toarray()
    vocab = vectorizer.get_feature_names()
    k = list(bodies.keys())

    bodiesTokens = loadBodiesTokens(bodiesFile)

    with open(stancesFile, 'r', encoding='UTF-8') as csvDataFile1: 
		 
        csvReader1 = csv.reader(csvDataFile1)
        first = 1
        for row in csvReader1:
            f = []
            if first == 1: 
                first = 0
            else:
                print(count)
                count = count + 1

                #class
                classes.append(keys[row[2]])	

                #canberra distance
                f.append(feat.canberraDist(row[0],bodies[row[1]], bigram_vectorizer))
                         
                #polarity scores
                neg, neu, pos = feat.polarityScores(row[0], bodies[row[1]])
                f.append(neg)
                f.append(neu)
                f.append(pos)

                tokens1 = tknzr.tokenize(row[0])
                tokens1=[token.lower() for token in tokens1 if (token.isalpha() and token not in stop_words)]
                tokens2 = bodiesTokens[row[1]]

                #word movers distance
                f.append(feat.wmd(tokens1, tokens2,wmd_model))

                #common words
                common = (set(tokens1) & set(tokens2))              
                f.append(feat.overlap(common))      
                        
                #tfidf
                f.append(feat.tfidf(tfidfMat, common,vocab,k.index(row[1])))
                               
                #negations
                f.append(feat.negWords(tokens1,tokens2))

                #add all features
                features.append(f)
								
    return np.array(features), np.array(classes)