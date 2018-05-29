import math
import os
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import nltk.sentiment.vader as vader
import numpy as np
from scipy.spatial.distance import canberra
#from gensim.models import KeyedVectors
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')

def canberraDist(h,b, bigram_vectorizer):
   
    corpus = []
    corpus.append(h)
    corpus.append(b)
    vecs = bigram_vectorizer.fit_transform(corpus).toarray()
    vec1, vec2 = vecs[0,:],vecs[1,:]
    #print (canberra(vec1,vec2))
    return canberra(vec1,vec2)/1000


def tfidf(mat, common, vocab, doc):
    tfidf = 0
    for w in common:
        if w in vocab:
            tfidf = tfidf + mat[doc,vocab.index(w)]
    return tfidf


def overlap(common):
    return len(common)
    

def wmd(headline, body,wmd_model):
    dist =  wmd_model.wmdistance(headline,body)
    if math.isinf(dist) or math.isnan(dist):
        return 1.5
    else:
        return dist
    

def polarityScores(headline,body):
    sid = SIA()
    h = sid.polarity_scores(headline)
    b = sid.polarity_scores(body)
    return abs(h['neg'] - b['neg']), abs(h['neu'] - b['neu']) , abs(h['pos'] - b['pos'])
    #return h['pos'], b['pos']
           
def negWords(headline,body):
    b_count = 0
    h_count = 0
    for b in body:
        if vader.negated(b):
            b_count = b_count + 1
    for h in headline:
        if vader.negated(h):
            h_count = h_count + 1
    hlen = len(headline)
    blen = len(body)
    if hlen!=0 and blen!=0:
        return abs((b_count/blen)-(h_count/hlen))
    else:
        #print("------------------------------------------------------")
        return 0
    



