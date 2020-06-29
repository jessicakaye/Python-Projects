# Spotify_Preprocessing.py
# 4/19/20
# @jessicakaye
# Used to process artist info for tracks for popularity classification dataset and create popularity class label


import json, csv, sys
import pandas as pd
from ast import literal_eval
import datetime
import statistics

spotify_df = pd.read_csv('spotify_data 3-22.csv')
spotify_df=spotify_df.filter(['album','artists','duration_ms', 'id', 'name','popularity','genre', 'danceability',
                              'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness','instrumentalness',
                              'liveness', 'valence', 'tempo', 'time_signature'])

spotify_df.loc[:,'artists'] = spotify_df.loc[:,'artists'].apply(lambda x: literal_eval(x))
spotify_df.loc[:,'album'] = spotify_df.loc[:,'album'].apply(lambda x: literal_eval(x))
spotify_df['primary_artist'] = ""
spotify_df['primary_artist_popularity'] = ""
spotify_df['contains_features?'] = 0
spotify_df['ft_artist'] = ""
spotify_df['max_ft_artist_popularity'] = 999
spotify_df['refined_artist_popularity'] = ""
spotify_df['album_release_date'] = ""
spotify_df['album_release_date_precision'] = ""
spotify_df['days_since_release'] = ""
spotify_df['album_type'] = ""

print(spotify_df.head())

for track in spotify_df.index:
    spotify_df.loc[track, 'primary_artist'] = spotify_df.loc[track, 'artists'][0]['name']
    spotify_df.loc[track, 'primary_artist_popularity'] = spotify_df.loc[track, 'artists'][0]['popularity']
    spotify_df.loc[track, 'album_type'] = spotify_df.loc[track, 'album']['album_type']
    spotify_df.loc[track, 'album_release_date'] = spotify_df.loc[track, 'album']['release_date']
    spotify_df.loc[track, 'album_release_date_precision'] = spotify_df.loc[track, 'album']['release_date_precision']
    if spotify_df.loc[track, 'album_release_date_precision'] == 'day':
        spotify_df.loc[track, 'days_since_release'] = (datetime.datetime.strptime('2020-03-22' , '%Y-%m-%d') - \
            datetime.datetime.strptime(spotify_df.loc[track, 'album_release_date'] , '%Y-%m-%d')).days
    elif spotify_df.loc[track, 'album_release_date_precision'] == 'month':
        spotify_df.loc[track, 'days_since_release'] = (datetime.datetime.strptime('2020-03-22' , '%Y-%m-%d') - \
                                                      datetime.datetime.strptime(
                                                          spotify_df.loc[track, 'album_release_date'], '%Y-%m')).days
    else:
        spotify_df.loc[track, 'days_since_release'] = (datetime.datetime.strptime('2020-03-22' , '%Y-%m-%d') - \
                                                      datetime.datetime.strptime(
                                                          spotify_df.loc[track, 'album_release_date'], '%Y')).days
    if len(spotify_df.loc[track, 'artists']) > 1:
        max_a_pop = -1
        spotify_df.loc[track, 'contains_features?'] = 1
        for artist in range(1, len(spotify_df.loc[track, 'artists'])):
            if spotify_df.loc[track, 'artists'][artist]['popularity'] > max_a_pop:
                max_a_pop = spotify_df.loc[track, 'artists'][artist]['popularity']
                max_artist = spotify_df.loc[track, 'artists'][artist]['name']
        spotify_df.loc[track, 'max_ft_artist_popularity'] = max_a_pop
        spotify_df.loc[track, 'ft_artist'] = max_artist

    if spotify_df.loc[track, 'primary_artist_popularity'] > spotify_df.loc[track, 'max_ft_artist_popularity']:
        spotify_df.loc[track, 'refined_artist_popularity'] =  spotify_df.loc[track, 'primary_artist_popularity']
    else:
        spotify_df.loc[track, 'refined_artist_popularity'] = statistics.mean([spotify_df.loc[track,
                                                            'primary_artist_popularity'], spotify_df.loc[track,
                                                            'max_ft_artist_popularity']])
spotify_df['is_single'] = 0
spotify_df['is_compilation'] = 0
spotify_df['is_album'] = 0

spotify_df.loc[spotify_df['album_type'] == 'single', 'is_single'] = 1
spotify_df.loc[spotify_df['album_type'] == 'compilation', 'is_compilation'] = 1
spotify_df.loc[spotify_df['album_type'] == 'album', 'is_album'] = 1

spotify_df['popularity_class_label'] = 'Not Popular'
spotify_df.loc[spotify_df['popularity'] >= spotify_df.loc[:,'popularity'].median(), 'popularity_class_label'] = 'Popular'


spotify_df.to_csv('processed_spotify_data_3-22.csv', index=False)
