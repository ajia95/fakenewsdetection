Dependencies:
Install the following libraries
-Python 3
-numpy
-scipy
-nltk
-sklearn
-matplotlib
-gensim

Download GoogleNews-vectors-negative300.bin from:

https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit


Structure:

data mining
----data
--------GoogleNews-vectors-negative300.bin 
--------testing
------------competition_test_bodies.csv
------------competition_test_stances.csv
--------training
------------train_bodies.csv             
------------train_stances.csv		(used by split.py)
------------training_stances.csv	(created by split.py)
------------validation_stances.csv	(created by split.py)
----collection.py
----cosSim.py
----data.py
----featureImportance.py
----features.py
----klDiv.py
----linearRegression.py
----logisticRegression.py
----models.py
----readme.txt



split.py

-run this to split the train_stances.csv file into training_stances.csv and 
validation_stances.csv

cosSim.py

-run this to calculate vector representations and their cosine similarity

klDiv.py

-run this to calculate kl divergence

featureImportance.py

-run this to calculate the features and print our their importance scores

models.py

-run this to compute the data features, train both models and test both models. 
-Trains and tests the linear regression model first, printing out the accuracy 
and F1 scores
-Trains and tests the logistic regression model second, also printingg out accurcy
F1 scores

features.py

-contains the code for each feature implemented

data.py

-used to compute the features and output classes for the datasets

linearRegression.py

-code for linear regression model

logisticRegression.py

-code for logistic regression model

collection.py

-contains functions used by other files



