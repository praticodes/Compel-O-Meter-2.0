""" CSC111 Winter 2023 Course Project : Compel-O-Meter

Description
===========
This file contains a simulation function which can be used to get a Compellingness report for different tweets and a
tweet function to get tweets from a user's twitter handle.


Copyright
==========
This file is Copyright (c) 2023 Akshaya Deepak Ramachandran, Kashish Mittal, Maryam Taj and Pratibha Thakur
"""
import os
import pandas as pd
import analysis
# our graphical user interface can be found under the python file gui.py and gui_ai.py


def tweet(usernames: list) -> dict[str: list[str]]:
    """ This function scrapes a user's Twitter tweets from the internet and returns a dictionary with the
    username and their tweets as key-value pairs.

    Preconditions:
    - usernames != []

    """
    total_tweets = {}

    for username in usernames:
        # For a given user, generate a JSON file.
        os.system(f"snscrape --jsonl --max-results 100 twitter-search 'from:{username}'> aoc-tweets.json")
        tweets_df = pd.read_json('aoc-tweets.json', lines=True)

        # Access the tweet's written content.
        tweets = list(tweets_df.content)
        # Retain the words in the tweet. Remove unnecessary characters.
        for i in range(len(tweets)):
            tweets[i] = tweets[i].replace('\n', "")

        total_tweets[username] = tweets

    return total_tweets


def simulation() -> None:
    """ This fuction returns a compellingness report for the tweets made by certain twitter handles."""
    # these are twitter handles, feel free to change them!
    tweet_handles = ['taylorswift13', 'Cobratate']
    dict_of_tweet = tweet(tweet_handles)
    compelligness_without_ai = {}
    for handle in dict_of_tweet:
        result = analysis.compellingness_with_description((dict_of_tweet[handle][0]))
        compelligness_without_ai[handle] = result
    compelligness_with_ai = {}
    for handle in dict_of_tweet:
        result = analysis.compellingness_description_ai((dict_of_tweet[handle][0]))
        compelligness_with_ai[handle] = result
    print('Compellingness Reports Without AI')
    for handle in compelligness_without_ai:
        print(handle + ' ' + 'Compellingness Report :')
        print(compelligness_without_ai[handle][0] + '\n' + compelligness_without_ai[handle][1] + '\n'
              + compelligness_without_ai[handle][2] + '\n' + compelligness_without_ai[handle][3] + '\n'
              + compelligness_with_ai[handle][4] + '\n' + compelligness_with_ai[handle][5] + '\n')
        print('======================================================================')
    print('Compellingness Reports With AI')
    for handle in compelligness_without_ai:
        print(handle + ' ' + 'Compellingness Report :')
        print(compelligness_with_ai[handle][0] + '\n' + compelligness_with_ai[handle][1] + '\n'
              + compelligness_with_ai[handle][2] + '\n' + compelligness_with_ai[handle][3] + '\n'
              + compelligness_with_ai[handle][4] + '\n' + compelligness_with_ai[handle][5] + '\n')
        print('======================================================================')
