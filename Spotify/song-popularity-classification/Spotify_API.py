# Spotify_API.py
# 3/22/20
# @jessicakaye
# Used to pull tracks, track features, and artist popularity from Spotify API based on list of genres.

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import json, csv, sys
from pprint import pprint

from typing import Any, Union

# Client Credentials Flow
# NOTE: Enviornmental Variables were defined for the SPOTIPY_CLIENT_ID & SPOTIPY_CLIENT_SECRET
client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# open the csv for writing
data_file = open('spotify_data.csv', 'w', encoding='utf8', newline='')

# create the csv writer object
csv_writer = csv.writer(data_file)

#counter variable used for writing headers to CSV
count = 0

pop_query = 'genre:pop'
pop_test = sp.search(q=pop_query, limit=1, offset=0, type='track')
pop_test['tracks']['items'][0].update({'genre': 'pop'})
test_features = sp.audio_features(tracks=[pop_test['tracks']['items'][0]['id']])
pop_test['tracks']['items'][0].update(test_features[0])

for i in range(len(pop_test)):
    if count == 0:
        # Write Headers for CSV
        header = pop_test['tracks']['items'][i].keys()
        csv_writer.writerow(header)
        count += 1

genres = ['genre:pop', 'genre:rap', 'genre:rock', 'genre:latin', 'genre:hip hop', 'genre:trap']
for genre in genres:
    # Will take 2000 tracks per genre
    o_list = (list(range(0, 2000, 50)))
    for o in o_list:
        pop_tracks = sp.search(q=genre, limit=50, offset=o, type='track')
        for track in range(len(pop_tracks['tracks']['items'])):
            # Here we add the song genre as a column
            pop_tracks['tracks']['items'][track].update({'genre': genre.split(":", 1)[1]})
            # Here we add the features for each track
            pop_features = sp.audio_features(tracks=[pop_tracks['tracks']['items'][track]['id']])
            pop_tracks['tracks']['items'][track].update(pop_features[0])
            # We should also add the popularity of each artist HERE
            for each_artist in range(len(pop_tracks['tracks']['items'][track]['artists'])):
                pop_artist = sp.artist(pop_tracks['tracks']['items'][track]['artists'][each_artist]['id'])
                pop_tracks['tracks']['items'][track]['artists'][each_artist]['popularity'] = pop_artist['popularity']
            # Write data of CSV file
            csv_writer.writerow(pop_tracks['tracks']['items'][track].values())

data_file.close()

