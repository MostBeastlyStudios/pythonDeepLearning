import numpy as np
from sklearn import preprocessing, cross_validation, svm
import pandas as pd

#using sklearn's SVM for ease

df = pd.read_csv('data/breast-cancer-wisconsin.data.txt') #get data
df.replace('?',-99999, inplace=True) #get rid of missing data and give it an outlier
df.drop(['id'], 1, inplace=True) #drop the id column because it's irrelivant

X = np.array(df.drop(['class'],1)) #features should be everything except the class column
y = np.array(df['class']) #labels should be the class column

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2) #get the training and testing data

clf = svm.SVC() #create a object of the KNN algorithm class
clf.fit(X_train, y_train) #train the KNN

accuracy = clf.score(X_test, y_test) #get it's final accuracy
print(accuracy)

example_measures = np.array([4,2,1,1,1,2,3,2,1]) #create fake data
example_measures = example_measures.reshape(1,-1) #refit to say it's one set of data
prediction = clf.predict(example_measures) #get the prediction through the KNN
print(prediction)
