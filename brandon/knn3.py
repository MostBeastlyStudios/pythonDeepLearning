import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import pandas as pd
import random
style.use('fivethirtyeight')

#Using cancer data on our own algorithm

def k_nearest_neighbors(data,predict,k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups! Idiot.')
    distances = []
    for group in data: #for each group
        for features in data[group]: #look at each point
            #euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2)) #the real way
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict)) #numpy's way of making the calculation easier
            distances.append([euclidean_distance, group]) #add the distance to point

    votes = [i[1] for i in sorted(distances)[:k]] #array of group keys equal to the 3 smallest distances
    print(Counter(votes).most_common(1)) #print the most common of the three types
    vote_results = Counter(votes).most_common(1)[0][0] #make the group of the new point the same as the most common close points
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_results, confidence

df = pd.read_csv("data/breast-cancer-wisconsin.data.txt") #get file
df.replace('?', -99999, inplace=True) #replace ?
df.drop(['id'], 1, inplace = True) #drop id column
full_data = df.astype(float).values.tolist() #convert to list of lists of floats
random.shuffle(full_data) # shuffle the data

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))] #train data is first 80% of the data
test_data = full_data[-int(test_size*len(full_data)):] #test data is the last 20% of data

for i in train_data: #for each object in training data
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        total += 1
        if group == vote:
            correct += 1
        else:
            print(confidence)

print('Accuracy:', correct/total)


#for i in dataset: #display on graph
#    for ii in dataset[i]:
#        plt.scatter(ii[0],ii[1], s=100, color=i)

#plt.scatter(new_features[0], new_features[1], s=100)
#plt.show()
