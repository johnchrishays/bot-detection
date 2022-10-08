import re
import pandas as pd

def remove_trailing_zero(id_):
    pattern = "([0-9]+)\\s+0"
    s = re.search(pattern, id_)
    if s:
        return s.group(1)
    return id_

id_path = PROJ_PATH + "/data/varol-2017.dat"
is_varol=True

id_df = pd.read_csv(id_path, sep='\t', names=("id_", "class"))

#def get_botometer_dataset_ids()
id_path = PROJ_PATH + "/data/cresci-stock-2018.tsv"
id_df = pd.read_csv(id_path, sep="\t", names=['id_', 'class'])
id_df.loc[:, 'id_'] = id_df['id_'].astype(str)

golbeck = pd.read_csv(PROJ_PATH + "/data/golbeck/AllScreennames.txt", names=('id_', 'screenname'))
golbeck['id_'] = golbeck['id_'].astype(str)
if is_varol:
    id_df['id_'] = id_df['id_'].map(remove_trailing_zero)
    ids = set.intersection(set(id_df['id_']), set(golbeck['id_']))
    avail = id_df.loc[id_df['id_'].isin(ids)]
    if is_varol:
        avail.loc[:, 'class'].fillna(0, inplace=True)
        labels = avail['class']
        screennames = golbeck[golbeck['id_'].isin(ids)]

not_found = 0
data_path = PROJ_PATH  + "/data/golbeck/tweets/"

for row in screennames.to_dict(orient='records'):
    tweets_path = data_path + f"{row['screenname']}_tweets.csv"
    print(tweets_path)
    if not os.path.exists(tweets_path):
        not_found += 1

kantepe_path = PROJ_PATH + "/data/kantepe-2017/user.bson"

with open(kantepe_path,'rb') as f:
    data = bson.decode_all(f.read())
    kantepe = pd.DataFrame(data)

kaiser = pyreadr.read_r(PROJ_PATH + '/data/kaiser/data_botometer.RData')['data_botometer']

# EMD
ersahin_2017, ersahin_2017_labels = load_emd(PROJ_PATH + '/data/EMD-2017/Twitter_dataset.arff') # Load data
