import pandas as pd
import quandl #for getting data
import numpy as np
import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

df = quandl.get('WIKI/GOOGL') # get stock data

#print(df.head()) #raw data

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]] #format data
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) #fill empty data holes
forecast_out = int(math.ceil(0.1*len(df))) #get the number of days equal to 10% of the total days of the stock's life

df['label'] = df[forecast_col].shift(-forecast_out) #make 'label' column euqal to the forecast_col shifted by the number of days we want to see

X = np.array(df.drop(['label'],1)) #create features variable with everything except the label column
X = preprocessing.scale(X) #scale the feature data (not really sure what this is)
X_lately = X[-forecast_out:] #X = X until the point of -forecast_out
X = X[:-forecast_out:]

df.dropna(inplace=True) #drop NaN and nil values
y = np.array(df['label']) #create labels using the label column

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2) # 20% of the data, we're going to use as testing data. Shuffles features and labels and ouputs X_train, y_train and X_test and y_test

clf = LinearRegression() #create algorithm object (can be svm.SVR()) <----- purpose of script is to show how easily you can switch algorithms
clf.fit(X_train,y_train) #send in training data
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test) #get the accuracy

forecast_set = clf.predict(X_lately) #get the prediction using X_lately
print(forecast_set, accuracy, forecast_out) #print that data

df['Forecast'] = np.nan #create an empty set as a NaN array

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set: #for each object in the array
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
