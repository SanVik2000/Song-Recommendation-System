#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import missingno as ms

#function to convert literas to foat data-type
def convert_to_float(df, columns):
    for i in columns:
        df[i] = df[i].astype('float')
    return df

#function to combine two-rows of a pandas dataframe
def combine_two_rows(df):
    columns = list(df.columns)
    for i in range(0, len(df.columns)):
        columns[i] = columns[i] + " " + df.iloc[0, i]
    return columns

#function to combine header with first row of pandas dataframe
def combine_one_row(df):
    columns = list(df.columns)
    for i in range(0, len(df.columns)):
        if i == 0:
            columns[i] = df.iloc[0, i]
        else:
            columns[i] = columns[i] + " " + df.iloc[0, i]
    return columns

#function to format stings as decribed below
def format_strings(x):
    if '-' in x:
        return ''.join(x.split('-'))
    if x.find('/'):
        return '|'.join(x.split('/'))
    return x

#function to modify strings as decribed below
def modifyString(serie, val):
    for i in range(0, val):
        if serie[i] == 'Old-Time / Historic':
            serie[i] = 'OldTime|Historic'
    return serie


#reading csv files using pandas
echonest = pd.read_csv('../Data/fma_metadata/echonest.csv')
features = pd.read_csv('../Data/fma_metadata/features.csv')
genres = pd.read_csv('../Data/fma_metadata/genres.csv')
tracks = pd.read_csv('../Data/fma_metadata/tracks.csv')

print("================================================================================================================================")
print("Processing Echonest Dataset :")
#Processing echonest dataset
echonest.drop(['echonest.8', 'echonest.9', 'echonest.15', 'echonest.16', 'echonest.17', 'echonest.18', 'echonest.19'], axis=1, inplace=True)
echonest.drop(['echonest.10', 'echonest.11', 'echonest.12'], axis=1, inplace=True)
echonest.drop(0, axis=0, inplace=True)
echonest.iloc[0, 0] = echonest.iloc[1, 0]
echonest.drop(2, axis=0, inplace=True)
echonest.columns = echonest.iloc[0]
echonest.drop(1, axis=0, inplace=True)
echonest.reset_index(inplace=True)
echonest.drop('index', inplace=True, axis=1)

#converting all entries of echonest dataset except for the columns of track_id, artist_name and track_name into float
echonest = convert_to_float(echonest, set(echonest.columns) - set(['track_id', 'artist_name', 'release']))

print("Done Processing Echonest Dataset :")
print("================================================================================================================================")


#glancing through processed "ECHONEST" dataset
print("================================================================================================================================")
print("Processed Echonest Dataset :")
print(echonest.head())
print("================================================================================================================================")

print("================================================================================================================================")
print("Processing Features Dataset :")
#Processing features dataset
features.iloc[0,0] = features.iloc[2, 0]
features.drop(2, inplace=True)
features.columns = combine_two_rows(features)
features.drop([0, 1], inplace=True)
features.reset_index(inplace=True)
features.drop('index', axis=1, inplace=True)
features = features.astype(dtype='float')
features['feature track_id'] = features['feature track_id'].astype('int')

print("Done Processing Features Dataset :")
print("================================================================================================================================")


#glancing through processed "FEATURES" dataset
print("================================================================================================================================")
print("Processed Features Dataset :")
print(features.head())
print("================================================================================================================================")


print("================================================================================================================================")
print("Processing Tracks Dataset :")
#Processing tracks dataset
tracks.iloc[0,0] = tracks.iloc[1, 0]
tracks.drop(1, axis=0, inplace=True)
tracks.columns = combine_one_row(tracks)
tracks.drop(0, inplace=True)
tracks.reset_index(inplace=True)
tracks.drop(['index'], axis=1, inplace=True)
tracks.drop(['track.12 language_code', 'album.12 type'], axis=1, inplace=True)
tracks.drop('track.9 genres_all', axis=1, inplace=True)

def getList(cd):
    return cd[1:-1].split(',')

for i in range(0, 106574):
    if type(tracks['track.7 genre_top'][i]) == float:
        genre_list = getList(str(tracks['track.8 genres'][i]))
        count = len(genre_list)
        title = ""
        for j in range(0, count):
            title = title + str(genres['title'][j]) + str('|')
        tracks['track.7 genre_top'][i] = title

print("Done Processing Tracks Dataset :")
print("================================================================================================================================")


#glancing through processed "TRACKS" dataset
print("================================================================================================================================")
print("Processed Tracks Dataset :")
print(tracks.head())
print("================================================================================================================================")


print("Combining all datasets into a single entity")

features.columns = ['track_id'] + list(features.columns[1:])
echonest['track_id'] = echonest['track_id'].astype('int')
tracks['track_id'] = tracks['track_id'].astype('int')
features.sort_values(by='track_id', inplace=True)
tracks.sort_values(by='track_id', inplace=True)
echonest.sort_values(by='track_id', inplace=True)

count = 0
for i in range(0, 106574):
    if features['track_id'][i] == tracks['track_id'][i]:
        count += 1
    else:
        print(features['track_id'][i], tracks['track_id'][i])

final = pd.concat([features, tracks.drop('track_id', axis=1)], axis=1)

#glancing through combines "FINAL" dataset
print("================================================================================================================================")
print("Final Dataset :")
print(final.head())

#further feature engineering
final['track.7 genre_top'] = modifyString(final['track.7 genre_top'], 13129)
final['track.7 genre_top'] = final['track.7 genre_top'].apply(format_strings)

track_title = pd.DataFrame(tracks['track.19 title'])
track_title['track_id'] = tracks['track_id']

#generating metadata

print("Generating Metadata")
metadata = pd.DataFrame()
metadata['track_id'] = final['track_id']
track_title = track_title.set_index('track_id')
track_title.index = [int(i) for i in track_title.index]
metadata['album_title'] = final['album.10 title']
metadata['artist_name'] = final['artist.12 name']
metadata['genre'] = final['track.7 genre_top']
metadata = metadata.set_index('track_id')
metadata['track_title'] = track_title.loc[metadata.index]['track.19 title']

final.drop('album.10 title', axis=1, inplace=True)
final.drop('artist.12 name', axis=1, inplace=True)
final.drop('set split', axis=1, inplace=True)

genre_dummy = pd.DataFrame(data= np.zeros((13129, 163)), columns= list(genres['title'].unique()))
genre_list = pd.Series(data= genre_dummy.columns)
genre_list = modifyString(genre_list, 163)
genre_list = genre_list.apply(format_strings)
genre_dummy.columns= genre_list
genre_list = list(genre_list)


for i in range(0, 13129):
    if '|' in final['track.7 genre_top'][i]:
        divided_list = str(final['track.7 genre_top'][i]).split('|')
        count = len(divided_list)
        for j in range(0, count):
            if divided_list[j] in genre_list:
                location = genre_list.index(divided_list[j])
                genre_dummy.iloc[i, location] = 1
    else:
        location = genre_list.index(final['track.7 genre_top'][i])
        genre_dummy.iloc[i, location] = 1

final.drop(['track.7 genre_top'], axis= 1, inplace= True)

final = pd.concat([final, genre_dummy], axis= 1)

print("================================================================================================================================")
print("Final Processed Data : " )
print(final.head())
print("Metadata")
print(metadata.head())
print("================================================================================================================================")

#writing csv files
import os

if not os.path.isdir(os.path.join('datasets','final')):
    os.makedirs(os.path.join('datasets','final'))
    
metadata.to_csv('datasets/final/metadata.csv')
final.to_csv('datasets/final/final.csv')