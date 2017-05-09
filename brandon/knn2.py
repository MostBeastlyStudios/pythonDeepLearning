import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
style.use('fivethirtyeight')

#Using our own testing data

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

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

    return vote_results

result = k_nearest_neighbors(dataset, new_features, k=3) #get results
print(result)

for i in dataset: #display on graph
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1], s=100, color=i)

plt.scatter(new_features[0], new_features[1], s=100)
plt.show()
