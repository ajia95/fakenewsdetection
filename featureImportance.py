from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import data
import os

path = os.path.abspath("")

x,y = data.calcfeatures(path+'/data/training/training_stances.csv', path+'/data/training/train_bodies.csv')

clf = ExtraTreesClassifier()
clf = clf.fit(x, y)
print (clf.feature_importances_  )

#model = SelectFromModel(clf, prefit=True)
#x_new = model.transform(x)

#print (x_new)
