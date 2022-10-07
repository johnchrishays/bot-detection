""" Utilities for accessing datasets and returning them, ready for analysis. """
from ast import literal_eval
import csv
from datetime import datetime
from functools import reduce
import json
import os
import pandas as pd
import re
from scipy.io import arff
from sklearn.feature_extraction.text import CountVectorizer
import xml.etree.ElementTree as ET

from preprocess import load_json, preprocess_users, drop_and_one_hot, extract_users, COLUMNS_TO_DROP, DUMMY_COLUMNS


def load_twibot(path, drop_extra_cols=[]):
    """ Load twibot dataset. """
    with open(path) as f:
        twibot = json.load(f)
        twibot_labels = [int(ent['label']) for ent in twibot]
        profs = [ent['profile'] for ent in twibot]
        twibot_df = pd.DataFrame(profs)
    twibot_df["created_at"] = twibot_df["created_at"].apply(lambda dt: datetime.strptime(dt, "%a %b %d %H:%M:%S %z %Y ").timestamp())
    # Turn bool, string columns into ints
    for col in twibot_df:
        if (True in twibot_df):
            twibot_df[col] = twibot_df[col].astype(int)
        if (col in ['followers_count', 'friends_count', 'listed_count', 'favourites_count', 'statuses_count']):
            twibot_df[col] = twibot_df[col].astype(int)
        if (col in ['is_translation_enabled', 'has_extended_profile']):
            twibot_df[col] = twibot_df[col].astype(bool).astype(int) 
    drop_cols = COLUMNS_TO_DROP + ['profile_location', 'entities', 'id_str'] + drop_extra_cols
    twibot_one_hot = drop_and_one_hot(twibot_df, drop_cols, [ent for ent in DUMMY_COLUMNS + ['is_translator', 'contributors_enabled'] if ent not in drop_cols])
    return twibot_df, twibot_one_hot, twibot_labels

def load_bot_repo_dataset(data_path, labels_path):
    """ Load any dataset that comes from bot repo in standard form. """
    profs = extract_users(data_path)
    df, one_hot, labels = preprocess_users(profs, labels_path)
    return df, one_hot, labels 


def load_cresci(data_template, folder_names, is_bot, cols_to_drop, dummy_cols, include_created_at, balance=False):
    """ Load one of the cresci datasets. """
    dfs = []
    cresci_labels = []

    for name, ib in zip(folder_names, is_bot):
        df = pd.read_csv(data_template.format(name))
        dfs.append(df)
        cresci_labels.extend([ib]*len(df))
        
    if balance:
        n_accts = min(map(len, dfs))
        dfs = [df.sample(n_accts) for df in dfs]
        cresci_labels = list(reduce(lambda a,b: a+b, [[ib]*n_accts for ib in is_bot]))

    cresci = pd.concat(dfs)

    if include_created_at:
        cresci["created_at"] = cresci["created_at"].apply(lambda dt: datetime.strptime(dt, "%a %b %d %H:%M:%S %z %Y").timestamp())
    cresci_labels = pd.Series(cresci_labels)

    # Preprocess
    cresci_one_hot = drop_and_one_hot(cresci, cols_to_drop, dummy_cols)
    return cresci, cresci_one_hot, cresci_labels

def load_cresci2017(data_template):
    """ Load cresci 2017 dataset. """
    # Load in data
    folder_names = ['fake_followers', 
    'genuine_accounts', 
    'social_spambots_1', 
    'social_spambots_2', 
    'social_spambots_3', 
    'traditional_spambots_1', 
    'traditional_spambots_2', 
    'traditional_spambots_3',
    'traditional_spambots_4']
    is_bot = [1, 0, 1, 1, 1, 1, 1, 1, 1]
    cols_to_drop = COLUMNS_TO_DROP + ['profile_banner_url', 
                                        'test_set_1', 
                                        'test_set_2', 
                                        'crawled_at',
                                        'updated', 
                                        'timestamp',
                                        'following', 
                                        'follow_request_sent',
                                        'created_at'
                                      ]
    dummy_cols = DUMMY_COLUMNS + ['is_translator', 'contributors_enabled', 'notifications']
    return load_cresci(data_template, folder_names, is_bot, cols_to_drop, dummy_cols, False)


def load_gilani_derived_bands(data_path):
    """ 
    Load gilani-2017 dataset of features derived in the original paper. 
    Split into bands based on how many followers the account had. 
    """
    sizes = ['1k', '100k', '1M', '10M']

    dfs = []
    labels = []

    for s in sizes:
        human_path = data_path + f"humans/humans.{s}.csv"
        bot_path = data_path + f"bots/bots.{s}.csv"
        human_df = pd.read_csv(human_path)
        bots_df = pd.read_csv(bot_path)
        for band in sizes:
            if band == s:
                human_df[f'band_{band}'] = 1
                bots_df[f'band_{band}'] = 1
            else:
                human_df[f'band_{band}'] = 0
                bots_df[f'band_{band}'] = 0
        combined_df = pd.concat([human_df, bots_df])
        
        combined_df['source_identity_list'] = combined_df['source_identity'].apply(lambda s: s[1:-1].split(sep=';'))
        for i in range(7):
            combined_df[f'source{i}'] = combined_df['source_identity_list'].apply(lambda x: 1 if f'{i}' in x else 0)
        combined_df.drop(['screen_name', 'source_identity', 'source_identity_list'], axis=1, inplace=True)

        labels.append(pd.Series([0]*len(human_df) + [1]*len(bots_df)))
        dfs.append(combined_df)
    return dfs, labels


def load_gilani_derived_combined(data_template):
    """ Load each of the bands from gilani-2017 and return their concatinated results. """
    dfs, labels = load_gilani_derived_bands(data_template)
    return pd.concat(dfs), pd.concat(labels)


def load_cresci2015(data_template):
    """ Load cresci-2015 dataset. """
    folder_names = ["elzioni2013", "TheFakeProject", "intertwitter", "twittertechnology", "fastfollowerz"]
    is_bot = [0, 0, 1, 1, 1]
    cols_to_drop = ['id', 'lang', 'name', 'screen_name',
                    'url', 'profile_image_url',
                    'profile_banner_url',
                    'profile_background_image_url_https', 'profile_text_color',
                    'profile_image_url_https', 'profile_sidebar_border_color',
                    'profile_sidebar_fill_color',
                    'profile_background_image_url', 'profile_background_color',
                    'profile_link_color',
                    'description', 'dataset', 'updated']
    dummy_cols = [col for col in DUMMY_COLUMNS if col != 'lang']
    return load_cresci(data_template, folder_names, is_bot, cols_to_drop, dummy_cols, True)


def load_midterm(data_path, labels_path):
    """ Load midterm-2018 dataset. """
    profs = load_json(data_path)
    cols_to_drop = ['probe_timestamp', 'screen_name', 'name', 'description', 'url']
    dummy_cols = ['lang', 'protected', 'verified', 'geo_enabled', 'profile_use_background_image', 'default_profile', 'index']
    df, one_hot, labels = preprocess_users(profs, labels_path, 'user_id', 'user_created_at', "%a %b %d %H:%M:%S %Y", cols_to_drop, dummy_cols)
    return df, one_hot, labels

def load_caverlee(data_path, drop_created_at=False):
    """ Load caverlee-2011 dataset. Since all data is numeric, no one-hot columns. """
    caverlee2011_bots = pd.read_csv(data_path + "content_polluters.txt", sep="\t", header=None, names=["UserID", "CreatedAt", "CollectedAt", "NumerOfFollowings", "NumberOfFollowers", "NumberOfTweets", "LengthOfScreenName", "LengthOfDescriptionInUserProfile"])
    caverlee2011_humans = pd.read_csv(data_path + "legitimate_users.txt", sep="\t", header=None, names=["UserID", "CreatedAt", "CollectedAt", "NumerOfFollowings", "NumberOfFollowers", "NumberOfTweets", "LengthOfScreenName", "LengthOfDescriptionInUserProfile"])
    caverlee2011 = pd.concat([caverlee2011_bots, caverlee2011_humans])
    caverlee2011["CreatedAt"] = caverlee2011["CreatedAt"].apply(lambda dt: datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").timestamp())
    caverlee2011.drop(columns=["CollectedAt", "UserID"], inplace=True)
    if drop_created_at:
        caverlee2011.drop(columns=["CreatedAt"], inplace=True)
    caverlee2011_labels = pd.Series([1]*len(caverlee2011_bots) + [0]*len(caverlee2011_humans))
    return caverlee2011, caverlee2011_labels


def load_emd(data_path):
    """ Load EMD-2017 dataset. All data is numeric, so no one-hot columns needed. """
    # Load data
    data = arff.loadarff(data_path)
    emd = pd.DataFrame(data[0])

    for col in emd:
        if (type(emd[col][0]) == type(b'str')):
            emd[col] = emd[col].apply(bool)

    emd_labels = emd['is_fake'].apply(lambda x: 1 if x else 0)
    return emd, emd_labels

def get_tweets_xml(path):
    """ Get tweets from xml file. """
    with open(path) as xmlfile:
        tree = ET.parse(xmlfile)
        tweets = (document.text for document in tree.getroot()[0])
        tweets_joined = " ".join(tweets)
        print(len(tweets))
        return tweets_joined
    
def get_labels_pan19(path):
    """ Get labels from pan19 labels file. """
    d = dict()
    with open(path) as file:
        for line in file:
            labels = line.split(":::")
            d[labels[0]] = 1 if labels[1] == 'bot' else 0
    return pd.DataFrame.from_dict(d, orient='index', columns=['label'])['label']


def get_tweets_pan19(index, data_path_template):
    """ Get tweets from xml, turn into dictionary from user_id to tweets. """
    d = {ind : get_tweets_xml(data_path_template.format(ind)) for ind in index}
    return pd.DataFrame.from_dict(d, orient='index', columns=['tweets'])


def tweets_to_countvectorized_df(df):
    """
    Input dataframe with each row a single string with all tweets collected.
    """
    cv = CountVectorizer(stop_words='english', min_df=25) 
    cv_matrix = cv.fit_transform(df) 
    cv_df = pd.DataFrame(cv_matrix.toarray(), index=df.index, columns=cv.get_feature_names())
    return cv_df

def load_pan19(data_path_template, labels_path):
    """ Load pan19 dataset and transform into work frequency across all user tweets. """
    labels = get_labels_pan19(labels_path)
    tweets = get_tweets_pan19(labels.index, data_path_template)
    # Get count vectorizer df
    pan19_cv_df = tweets_to_countvectorized_df(tweets['tweets'])
    return pan19_cv_df.loc[labels.index], labels


def load_cresci_tweets(data_path_template, folder_names, is_bot):
    """ Load tweet data for cresci2017 and cresci2015. """
    tweets = []
    cresci_labels = []

    for name, ib in zip(folder_names, is_bot):
        df = pd.read_csv(data_path_template.format(name), encoding='latin-1')
        df['text'] = df['text'].apply(lambda x: "" if isinstance(x, float) else x)
        df_groups = df[['text', 'user_id']].groupby(['user_id'])
        print(df_groups.size())
        df_tweets = df_groups['text'].apply(lambda x: " ".join(x))
        tweets.append(df_tweets)
        cresci_labels.extend([ib]*len(df_tweets))
            
    cresci_tweets = pd.concat(tweets)
    cresci_cv_df = tweets_to_countvectorized_df(cresci_tweets)
    return cresci_cv_df, pd.Series(cresci_labels)


def load_cresci2017_tweets(data_path_template):
    """ Load text data for cresci2017. """
    folder_names = ['fake_followers', 
            'genuine_accounts', 
            'social_spambots_1', 
            'social_spambots_2', 
            'social_spambots_3', 
            'traditional_spambots_1']
    is_bot = [1, 0, 1, 1, 1, 1]
    return load_cresci_tweets(data_path_template, folder_names, is_bot)


def load_cresci2015_tweets(data_path_template):
    """ Load text data for cresci2015. """
    folder_names = ["elzioni2013", 
            "TheFakeProject", 
            "intertwitter", 
            "twittertechnology", 
            "fastfollowerz"]
    is_bot = [0, 0, 1, 1, 1]
    return load_cresci_tweets(data_path_template, folder_names, is_bot)


def load_cresci_stock_tweets(profiles_data_path, labels_path, tweets_path_template):
    """ Load tweets data from cresci_stock dataset. """
    profiles = extract_users(profiles_data_path)
    tweets_dict = {}

    for prof in profiles:
        screen_name = prof['screen_name']
        user_id = prof['id']
        tweets_path = tweets_path_template.format(screen_name)
        if os.path.exists(tweets_path):
            with open(tweets_path) as f:
                tweets = " ".join((literal_eval(line['text']).decode('utf-8') for line in csv.DictReader(f)))
                tweets_dict[user_id] = tweets

    cresci_stock_tweets = pd.DataFrame.from_dict(tweets_dict, orient='index', columns=['tweets'])
    labels = pd.read_csv(labels_path, sep="\t", header=None, names=["id", "label_str"], index_col="id")
    labels.loc[labels['label_str']=="human", 'label'] = 0
    labels.loc[labels['label_str']=="bot", 'label'] = 1
    cresci_stock_tweets, labels = cresci_stock_tweets.align(labels, join="inner", axis=0)
    cresci_stock_tweets.reset_index(inplace=True)
    labels = labels['label']
    cv_df = tweets_to_countvectorized_df(cresci_stock_tweets['tweets'])
    return cv_df, pd.Series(labels)


def process_line_yang(line):
    """ Replace python2 timestamp with python3-readable int, return tuple. """
    pattern = '([0-9]+)L'
    timestamp = re.search(pattern, line).group(1)
    line = re.sub(pattern, timestamp, line)
    tup = literal_eval(line)
    return tup


def get_yang_datafile(data_path, col_names):
    """ Open yang datafile and return dataframe for further processing. Either human or bot datafile. """
    with open(data_path) as f:
        lines = [process_line_yang(line) for line in f.readlines()]
    return pd.DataFrame(lines, columns=col_names)


def load_yang(data_path):
    """ Load yang-2013 dataset. """
    human_col_names = ["user_id", "background_url", "account creation time",  "description", "favourites_count", "followers_count", "followings_count", "geo_enabled", "location", "name", "image_url", "protected", "screen_name", "statuses_count", "timezone", "verified"]
    bot_col_names =  human_col_names + ["language", "list_count", "bio_url"]
    yang_humans = get_yang_datafile(data_path + "NormalData/users.txt", human_col_names)
    yang_bots = get_yang_datafile(data_path + "MalData/users.txt", bot_col_names)
    yang = pd.concat([yang_humans, yang_bots], join='inner')
    labels = pd.Series([0]*len(yang_humans) + [1]*len(yang_bots))
    yang_one_hot = drop_and_one_hot(yang, ['user_id', 'description', 'screen_name'], ['background_url', 'location', 'name', 'image_url', 'timezone'])
    return yang, yang_one_hot, labels
           

def load_yang_tweets(data_path):
    """ Load yang-2013 tweet data. """
    human_col_names = ['tweet_id', 'user_id', 'tweet creation time', 'favourited', 'in_reply_to_status_id', 'in_reply_to_user_id', 'in_reply_to_screen_name', 'latitude', 'longitude', 'source', 'retweet', 'truncated', 'text']
    bot_col_names = human_col_names + ['retweet_count', 'hash_tag', 'mention_users']
    yang_humans = get_yang_datafile(data_path + "NormalData/tweets.txt", human_col_names)
    yang_bots = get_yang_datafile(data_path + "MalData/tweets.txt", bot_col_names)
    human_groups = yang_humans[['text', 'user_id']].groupby(['user_id'])
    human_tweets = human_groups['text'].apply(lambda x: " ".join(x))
    bot_groups = yang_bots[['text', 'user_id']].groupby(['user_id'])
    bot_tweets = bot_groups['text'].apply(lambda x: " ".join(x))
    df = pd.concat([human_tweets, bot_tweets], join='inner')
    labels = pd.Series([0]*len(human_tweets) + [1]*len(bot_tweets))
    cv_df = tweets_to_countvectorized_df(df)
    return cv_df, pd.Series(labels)
        
        
