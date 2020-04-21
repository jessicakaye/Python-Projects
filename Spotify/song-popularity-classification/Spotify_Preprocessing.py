# Spotify_Preprocessing.py
# 4/19/20
# @jessicakaye
# Used to process artist info for tracks for popularity classification dataset and create popularity class label


import json, csv, sys
import pandas as pd
from ast import literal_eval

spotify_df = pd.read_csv('spotify_data 3-22.csv')
spotify_df=spotify_df.filter(['album','artists','duration_ms', 'id', 'name','popularity','genre', 'danceability',
                              'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness','instrumentalness',
                              'liveness', 'valence', 'tempo', 'time_signature'])

spotify_df.loc[:,'artists'] = spotify_df.loc[:,'artists'].apply(lambda x: literal_eval(x))
spotify_df['primary_artist'] = ""
spotify_df['primary_artist_popularity'] = ""
spotify_df['contains_features?'] = False
spotify_df['ft_artist'] = ""
spotify_df['max_ft_artist_popularity'] = ""
spotify_df['popularity_class_label'] = ""
spotify_df.loc[spotify_df['popularity'] >= spotify_df.loc[:,'popularity'].median(), 'popularity_class_label'] = 'Popular'
spotify_df.loc[spotify_df['popularity'] < spotify_df.loc[:,'popularity'].median(), 'popularity_class_label'] = 'Not Popular'


print(spotify_df.head())

for track in spotify_df.index:
    spotify_df.loc[track, 'primary_artist'] = spotify_df.loc[track, 'artists'][0]['name']
    spotify_df.loc[track, 'primary_artist_popularity'] = spotify_df.loc[track, 'artists'][0]['popularity']
    if len(spotify_df.loc[track, 'artists']) > 1:
        max_a_pop = -1
        spotify_df.loc[track, 'contains_features?'] = True
        for artist in range(1, len(spotify_df.loc[track, 'artists'])):
            if spotify_df.loc[track, 'artists'][artist]['popularity'] > max_a_pop:
                max_a_pop = spotify_df.loc[track, 'artists'][artist]['popularity']
                max_artist = spotify_df.loc[track, 'artists'][artist]['name']
        spotify_df.loc[track, 'max_ft_artist_popularity'] = max_a_pop
        spotify_df.loc[track, 'ft_artist'] = max_artist

spotify_df.to_csv('processed_spotify_data_3-22.csv', index=False)
