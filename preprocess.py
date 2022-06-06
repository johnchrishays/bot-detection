""" Utilities for preprocessing datasets. """
from datetime import datetime
import json
import pandas as pd

COLUMNS_TO_DROP = ['id', 
                   'name', 
                   'screen_name', 
                   'url', 
                   'profile_image_url', 
                   'profile_background_image_url_https', 
                   'description',
                   'profile_image_url_https', 
                   'profile_background_image_url', 
                   'profile_text_color', 
                   'profile_sidebar_border_color', 
                   'profile_sidebar_fill_color', 
                   'profile_background_color', 
                   'profile_link_color']
DUMMY_COLUMNS = ['lang', 
                  'time_zone', 
                  'location', 
                  'default_profile', 
                  'default_profile_image', 
                  'utc_offset',
                  'default_profile', 
                  'default_profile_image', 
                  'profile_background_tile', 
                  'utc_offset',
                  'protected', 
                  'geo_enabled',
                  'verified',
                  'profile_use_background_image']


def load_json(data_path):
    """
    Load json file and return.
    """
    with open(data_path) as f:
        d = json.load(f)
    return d


def extract_users(data_path):
    """
    Load json file, extract user dictionaries and return.
    """
    d = load_json(data_path)
    profs = [ent['user'] for ent in d]
    return profs

def drop_and_one_hot(X, drop_cols, one_hot_cols):
    """
    Drop columns listed by drop_cols, turn one_hot_cols into 
    one-hot variables since decision trees must be fitted on numerical data.
    """
    X.drop(columns=drop_cols, inplace=True)
    one_hot = pd.get_dummies(X, columns=one_hot_cols, drop_first=True)
    return one_hot

def preprocess_users(profs, labels_path, user_id_col='id', user_created_at_col='created_at', time_format="%a %b %d %H:%M:%S %z %Y", cols_to_drop=COLUMNS_TO_DROP + ['profile_banner_url', 'entities', 'id_str', 'following', 'follow_request_sent', 'notifications', 'translator_type'], dummy_cols=DUMMY_COLUMNS + ['is_translator', 'contributors_enabled', 'is_translation_enabled', 'has_extended_profile']):
    """
    Preprocessing user dictionaries.
    - Convert created_at to datetime
    - Read in labels
    - Drop some cols and turn others into one-hot
    - Return dataframe, one_hot dataframe and labels
    """
    df = pd.DataFrame(profs).set_index(user_id_col)
    df[user_created_at_col] = df[user_created_at_col].apply(lambda dt: datetime.strptime(dt, time_format).timestamp())
    labels = pd.read_csv(labels_path, sep="\t", header=None, names=["id", "label_str"], index_col="id")
    labels.loc[labels['label_str']=="human", 'label'] = 0
    labels.loc[labels['label_str']=="bot", 'label'] = 1
    df, labels = df.align(labels, join="inner", axis=0)
    df.reset_index(inplace=True)
    labels = labels['label']
    one_hot = drop_and_one_hot(df, cols_to_drop, dummy_cols)
    return df, one_hot, labels

