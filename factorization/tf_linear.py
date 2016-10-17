import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score


def get_data():
    df = pd.read_csv("/home/cully/returnpath/interview_data.csv").drop('id', axis=1).dropna()
    df.columns = ['_'.join(x.lower().split('-')) for x in df.columns]
    return df

df = get_data()
train, test = train_test_split(df, test_size=.2)

COLUMNS = ['_'.join(x.lower().split('-')) for x in df.columns]

df.columns = COLUMNS

print df.columns

LABEL_COLUMN = 'read_rate'

CATEGORICAL_COLUMNS = ['from_domain_hash', 'domain_extension', 'day']

CONTINUOUS_COLUMNS = [x for x in COLUMNS if x not in CATEGORICAL_COLUMNS and x != LABEL_COLUMN]


def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}

    categorical_cols = {k : tf.SparseTensor(
        indices=[[i,0] for i in range(df[k].size)],
        values=df[k].values,
        shape=[df[k].size, 1])
                        for k in CATEGORICAL_COLUMNS}

    feature_cols = dict(continuous_cols.items() + categorical_cols.items())

    label = tf.constant(df[LABEL_COLUMN].values)

    return feature_cols, label

def train_input_fn():
    return input_fn(train)

def test_input_fn():
    return input_fn(test)



#if __name__ == '__main__':

from_domain_hash = tf.contrib.layers.sparse_column_with_hash_bucket("from_domain_hash", hash_bucket_size=int(1e5))

domain_extension = tf.contrib.layers.sparse_column_with_hash_bucket("domain_extension", hash_bucket_size=int(1e3))

day = tf.contrib.layers.sparse_column_with_hash_bucket('day', hash_bucket_size=int(1e1))


campaign_size = tf.contrib.layers.real_valued_column('campaign_size')
unique_user_cnt = tf.contrib.layers.real_valued_column('unique_user_cnt')
avg_domain_read_rate = tf.contrib.layers.real_valued_column('avg_domain_read_rate')
avg_domain_inbox_rate = tf.contrib.layers.real_valued_column('avg_domain_inbox_rate')
avg_user_avg_read_rate = tf.contrib.layers.real_valued_column('avg_user_avg_read_rate')
avg_user_domain_avg_read_rate = tf.contrib.layers.real_valued_column('avg_user_domain_avg_read_rate')
mb_superuser = tf.contrib.layers.real_valued_column('mb_superuser')
mb_engper = tf.contrib.layers.real_valued_column('mb_engper')
mb_supersub = tf.contrib.layers.real_valued_column('mb_supersub')
mb_engsec = tf.contrib.layers.real_valued_column('mb_engsec')
mb_inper = tf.contrib.layers.real_valued_column('mb_inper')
mb_insec = tf.contrib.layers.real_valued_column('mb_insec')
mb_unengsec = tf.contrib.layers.real_valued_column('mb_unengsec')
mb_idlesub = tf.contrib.layers.real_valued_column('mb_idlesub')


model = tf.contrib.learn.LinearRegressor(feature_columns=[from_domain_hash, domain_extension, day, campaign_size, unique_user_cnt, avg_domain_read_rate, avg_domain_inbox_rate, avg_user_avg_read_rate, avg_user_domain_avg_read_rate, mb_superuser, mb_engper, mb_supersub, mb_engsec, mb_inper, mb_insec, mb_unengsec, mb_idlesub], model_dir='/home/cully/git/factorization/factorization/models')

model.fit(input_fn=train_input_fn, steps=2000)
results = model.evaluate(input_fn=test_input_fn, steps=1)

for key in sorted(results):
    print "Key: {0}, Value: {1}".format(key, results[key])

preds = model.predict(input_fn=test_input_fn)


test_2 = test.copy()

y_test = test_2.pop('read_rate').values
X_test = test_2.values

print r2_score(y_test, preds)




'''
opt = GradientDescentOptimizer(learning_rate=0.1)
opt_op=opt.minimize(cost, var_list=<list of variables>)

opt_op.run()
'''
