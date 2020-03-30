#importing libraries
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.utils import shuffle

#reading processed datasets
final = pd.read_csv('../Data/Processed/final.csv')
metadata = pd.read_csv('../Data/Processed/metadata.csv')

#X - train and Y - test
X = final.loc[[i for i in range(0, 6000)]]
Y = final.loc[[i for i in range(10000, final.shape[0])]]

metadata = metadata.set_index('track_id')

print("Train Data Shape : " , X.shape)
print("Test-Data Shape : "  , Y.shape)

#initialising sci-kit learn's k-means cluster
kmeans = KMeans(n_clusters=6)

# Fitting model
def fit(df, algo, flag=0):
    if flag:
        algo.fit(df)
    else:
         algo.partial_fit(df)          
    df['label'] = algo.labels_
    return (df, algo)

# Function to predict
def predict(t, Y):
    y_pred = t[1].predict(Y)
    mode = pd.Series(y_pred).mode()
    return t[0][t[0]['label'] == mode.loc[0]]

# Function to recommend which returns 1)Genre 2)Artist 3) Mixed
def recommend(recommendations, meta, Y):
    dat = []
    for i in Y['track_id']:
        dat.append(i)
    genre_mode = meta.loc[dat]['genre'].mode()
    artist_mode = meta.loc[dat]['artist_name'].mode()
    return meta[meta['genre'] == genre_mode.iloc[0]], meta[meta['artist_name'] == artist_mode.iloc[0]], meta.loc[recommendations['track_id']]

#fitting the k-means cluster
Recommend = fit(X, kmeans, 1)
#predicting the clusters
recommendations = predict(Recommend, Y.iloc[6:12])
#recommending the user the most similar ones based on 1) Genre 2) Artisy and 3) Mixed type of both Artist and Genre
output = recommend(recommendations, metadata, Y)


genre_recommend, artist_name_recommend, mixed_recommend = output[0], output[1], output[2]

print("================================================================================================================================")
print("Recommendation based on Genre")
print(genre_recommend.head())
print("================================================================================================================================")
print("Recommendation based on Artist")
print(artist_name_recommend.head())
print("================================================================================================================================")
print("Recommendation based on both Genre and Artists")
print(mixed_recommend.head())
