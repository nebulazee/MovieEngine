# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:09:28 2018

@author: DELL NOTEBOOK
"""
import operator
import pandas as pd
import numpy as np
from . import FindMostSimilar as fms
#reading user file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')
#print(users)
#print(users.shape)

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')
#print(ratings.shape)
#print(ratings)

#Reading items file:
i_cols = ['movie_id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')

#print(items.iloc[0]['movie_title'])

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
#training dataset
ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols,
 encoding='latin-1')
#print(ratings_train.shape)
#test dataset
ratings_test = pd.read_csv('ml-100k/ua.test',sep='\t',names=r_cols,encoding='latin-1')
#print(ratings_test.shape)

n_users = users.user_id.unique().shape[0]
n_items = items.movie_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in ratings_train.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
#print(train_data_matrix[0][0])
test_data_matrix = np.zeros((n_users, n_items))
for line in ratings_test.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]


from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

#print(user_similarity)
#print(item_similarity)
#print(user_similarity.shape[1])
#print(np.size(user_similarity,0))
#print(np.size(user_similarity,1))
#print(user_similarity[2][5])
#print(user_similarity[5][2])
#print(np.size(item_similarity,0))
#print(np.size(item_similarity,1))

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')
dic={}
for i in range(1680):
    dic[items.iloc[i]['movie_id']]=items.iloc[i]['movie_title']

id=1;
def inter(id1):
    id=1
    list=[]
    for closeness in item_prediction[id1]:
        list.append((closeness,id))
        id=id+1
    list.sort(key=operator.itemgetter(0),reverse=True)
    return list   
#Toy Story (1995)

#print(list)
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
#
def getRecommendation(input):
    for i in range(1682):
       # print(items.iloc[i]['movie_title'])
        if input==items.iloc[i]['movie_title']:
            return (items.iloc[i]['movie_id'])
   
def recommendMovies(movie):
    c=1;
    lMov={};
    movie=fms.findmostSimilarItem(movie)
    #i=getRecommendation('Exotica (1994)')
    i=getRecommendation(movie)
    print(i)
    mList=inter(i)
    for i in mList:
        if c==15:
            break
        lMov[c]=(dic[i[1]]+str(i[0]));
        print(dic[i[1]]+str(i[0]))
        c=c+1
    return lMov;