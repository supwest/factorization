import numpy as np
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer


def load_data(filename, user_data=None, item_data=None):
    data = []
    y = []
    users = set()
    items = set()
    item_keys = ['itemid', 'title', 'release_date', 'url', 'unknown', 'action', 'adventure', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 'musical', 'romance', 'sci_fi', 'thriller', 'war', 'western']
    if item_data is not None:
        with open(item_data) as f:
            for line in f:
                tmp = line.strip().split('|')
    with open(filename) as f:
        for line in f:
            (userid, movieid, rating, ts) = line.split('\t')
            #if item_data is not None:
            #    data.append({'user_id':str(userid), 'movie_id':str(movieid), 'action':str(item_data['action']), 'adventure':str(item_data['adventure'])})
            
            data.append({"user_id": str(userid), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(userid)
            items.add(movieid)

    return data, y, users, items


if __name__ == '__main__':
    filename = '/home/cully/Documents/galvanize/recommendation-systems/data/u.data'
    data, y, users, items = load_data(filename)

