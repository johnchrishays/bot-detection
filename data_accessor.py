""" Utilities for accessing datasets and returning them, ready for analysis. """
from datetime import datetime
import json
import pandas as pd
from scipy.io import arff
import xml.etree.ElementTree as ET

from preprocess import preprocess_users, drop_and_one_hot, extract_users, COLUMNS_TO_DROP, DUMMY_COLUMNS


def load_twibot(path, drop_extra_cols=[]):
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
    profs = extract_users(data_path)
    df, one_hot, labels = preprocess_users(profs, labels_path)
    return df, one_hot, labels 


def load_cresci(data_template, folder_names, is_bot, cols_to_drop, dummy_cols, include_created_at):
    """ Load one of the cresci datasets. """
    dfs = []
    cresci_labels = []

    for name, ib in zip(folder_names, is_bot):
        df = pd.read_csv(data_template.format(name))
        dfs.append(df)
        cresci_labels.extend([ib]*len(df))
        
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


def load_caverlee(data_path):
    """ Load caverlee-2011 dataset. Since all data is numeric, no one-hot columns. """
    caverlee2011_bots = pd.read_csv(data_path + "content_polluters.txt", sep="\t", header=None, names=["UserID", "CreatedAt", "CollectedAt", "NumerOfFollowings", "NumberOfFollowers", "NumberOfTweets", "LengthOfScreenName", "LengthOfDescriptionInUserProfile"])
    caverlee2011_humans = pd.read_csv(data_path + "legitimate_users.txt", sep="\t", header=None, names=["UserID", "CreatedAt", "CollectedAt", "NumerOfFollowings", "NumberOfFollowers", "NumberOfTweets", "LengthOfScreenName", "LengthOfDescriptionInUserProfile"])
    caverlee2011 = pd.concat([caverlee2011_bots, caverlee2011_humans])
    caverlee2011["CreatedAt"] = caverlee2011["CreatedAt"].apply(lambda dt: datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").timestamp())
    caverlee2011.drop(columns=["CollectedAt", "UserID"], inplace=True)
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
        return tweets_joined
    
def get_labels_pan19(path):
    """ Get labels from pan19 labels file. """
    d = dict()
    with open(path) as file:
        for line in file:
            labels = line.split(":::")
            d[labels[0]] = 1 if labels[1] == 'bot' else 0
    return pd.DataFrame.from_dict(d, orient='index', columns=['label'])

def get_tweets_pan19(index, data_path_template):
    d = {ind : get_tweets_xml(data_path_template.format(ind)) for ind in index}
    return pd.DataFrame.from_dict(d, orient='index', columns=['tweets'])

def load_pan19(data_path_template, labels_path):
    labels = get_labels_pan19(labels_path)
    tweets = get_tweets_pan19(labels.index, data_path_template)
    return tweets, labels


def load_cresci2017_tweets(data_path_template):
    """ Load text data for cresci2017. """
    # Load in data
    folder_names = ['fake_followers', 
    'genuine_accounts', 
    'social_spambots_1', 
    'social_spambots_2', 
    'social_spambots_3', 
    'traditional_spambots_1']
    is_bot = [1, 0, 1, 1, 1, 1]
    tweets = []
    cresci2017_labels = []

    for name, ib in zip(folder_names, is_bot):
        df = pd.read_csv(data_path_template.format(name), encoding='latin-1')
        df['text'] = df['text'].apply(lambda x: "" if isinstance(x, float) else x)
        df_groups = df[['text', 'user_id']].groupby(['user_id'])
        df_tweets = df_groups['text'].apply(lambda x: " ".join(x))
        tweets.append(df_tweets)
        print(name, len(df_tweets))
        cresci2017_labels.extend([ib]*len(df_tweets))
            
    cresci2017_tweets = pd.concat(tweets)
    return cresci2017_tweets, cresci2017_labels
