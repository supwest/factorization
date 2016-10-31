import numpy as np
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import train_test_split


def load_data(filename, user_data=None, item_data=None):
    data = []
    y = []
    users = set()
    items = set()
    item_keys = ['itemid', 'title', 'release_date', 'url', 'unknown', 'action', 'adventure', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 'musical', 'romance', 'sci_fi', 'thriller', 'war', 'western']
    #if item_data is not None:
    #    with open(item_data) as f:
    #        for line in f:
    #            tmp = line.strip().split('|')
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

def parse_item_data():
    filename='/home/cully/Documents/galvanize/recommendation-systems/data/u.item'
    item_data = []
    item_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            d = line.strip().split('|')
            item_dict[d[0]] = {'unknown':d[5], 'action':d[6], 'adventure':d[7], 'children':d[8], 'comedy':d[9], 'crime':d[10], 'documentary':d[11], 'drama':d[12], 'fantasy':d[13], 'film_noir':d[14], 'horror':d[15], 'musical':d[16], 'romance':d[17], 'sci_fi':d[18], 'thriller':d[19], 'war':d[20], 'western':d[21]}

            #item_data.append(item)
            
    return item_dict

def combine_item_data(data, item_data):
    for row in data:
        row.update(item_data[row['movie_id']])
    
    return data

if __name__ == '__main__':
    filename = '/home/cully/Documents/galvanize/recommendation-systems/data/u.data'
    data, y, users, items = load_data(filename)
    train_data, test_data, y_train, y_test = train_test_split(data, y, test_size=.2)

    item_data = parse_item_data()
    X_train = train_data
    X_test = test_data
    #X_train = combine_item_data(train_data, item_data)
    #X_test = combine_item_data(test_data, item_data)

    v = DictVectorizer()
    X_train = v.fit_transform(X_train)
    X_test = v.transform(X_test)
    fm = pylibfm.FM(num_factors=10, num_iter=10, verbose=True, task='regression', initial_learning_rate=0.001, learning_rate_schedule="optimal")

    fm.fit(X_train, y_train)

    preds = fm.predict(X_test)
    
    mse = mean_squared_error(preds, y_test)
    r2 = r2_score(preds, y_test)
    print mse, r2
