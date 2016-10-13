import tempfile
import urllib
import requests
import os
import tensorflow as tf
import pandas as pd


requests

def get_data(cols):
    train_file = tempfile.NamedTemporaryFile()
    test_file = tempfile.NamedTemporaryFile()
    label_col='label'
    if not os.path.exists('/home/cully/git/factorization/factorizaion/train_tf.csv'):
        with open('train_tf.csv', 'w') as f:
            f.write(requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data").text)
    if not os.path.exists('/home/cully/git/factorization/factorization/test_tf.csv/'):
        with open('test_tf.csv', 'w') as f:
            f.write(requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test").text)

    
    df_train = pd.read_csv('train_tf.csv', names=cols, skipinitialspace=True)
    df_test = pd.read_csv('test_tf.csv', names=cols, skipinitialspace=True, skiprows=1)
    df_train[label_col] = df_train['income_bracket'].apply(lambda x: ">50K" in x).astype(int)
    df_test[label_col] = df_test['income_bracket'].apply(lambda x: ">50K" in x).astype(int)
    #df_test = pd.read_csv('test_tf.csv')
    return df_train, df_test


def input_fn(df, cat_cols, cont_cols,label_col=None):
    continuous_cols = {k: tf.constant(df[k].values) for k in cont_cols}

    categorical_cols = {k: tf.SparseTensor(
        indices=[[i,0] for i in range(df[k].size)],
        values = df[k].values,
        shape = [df[k].size, 1]) for k in cat_cols}
    
    if label_col is not None:
        label = tf.constant(df[label_col].values)

    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    
    if label_col is not None:
        return feature_cols, label
    else:
        return feature_cols

#def train_input_fn(df, cat_cols, cont_cols, label_col=None):
#    return input_fn(df, cat_cols, cont_cols, label_col)
#
#def test_input_fn(df, cat_cols, cont_cols, label_col=None):
#    return input_fn(df, cat_cols, cont_cols, label_col)


if __name__ == '__main__':
    cols = ["age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race",
            "gender","capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income_bracket"] 

    df_train, df_test = get_data(cols)

    cat_cols = ['workclass', 'education', 'marital_status', 'occupation',
            'relationship', 'race', 'gender', 'native_country']
    cont_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    label_col = 'label'

    def train_input_fn():
        return input_fn(df_train, cat_cols, cont_cols, label_col)
    
    def test_input_fn():
        return input_fn(df_test, cat_cols, cont_cols, label_col)
    #train_data = train_input_fn(df_train, cat_cols, cont_cols)
    #test_data = test_input_fn(df_test, cat_cols, cont_cols)

    
    #make sparse tensors
    gender = tf.contrib.layers.sparse_column_with_keys(column_name='gender', keys=['Female', 'Male'])
    education = tf.contrib.layers.sparse_column_with_hash_bucket('education', hash_bucket_size=1000)

    race = tf.contrib.layers.sparse_column_with_keys(column_name='race', keys=['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'])
    marital_status = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='marital_status', hash_bucket_size=100)
    relationship = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='relationship', hash_bucket_size=100)
    workclass = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='workclass', hash_bucket_size=100)
    occupation = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='occupation', hash_bucket_size=1000)
    native_country = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='native_country', hash_bucket_size=1000)

    age = tf.contrib.layers.real_valued_column('age')
    education_num = tf.contrib.layers.real_valued_column('education_num')
    capital_gain = tf.contrib.layers.real_valued_column('capital_gain')
    capital_loss = tf.contrib.layers.real_valued_column('capital_loss')
    hours_per_week = tf.contrib.layers.real_valued_column('hours_per_week')

    age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18,25,30,35,40,45,50,55,60,65])

    education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))

    age_buckets_x_race_x_occupation = tf.contrib.layers.crossed_column([age_buckets, race, occupation], hash_bucket_size=int(1e6))

    model_dir = tempfile.mkdtemp()
    m = tf.contrib.learn.LinearClassifier(feature_columns=[gender, native_country, education, occupation, workclass, marital_status, race, age_buckets, education_x_occupation, age_buckets_x_race_x_occupation], model_dir=model_dir)

    m.fit(input_fn=train_input_fn, steps=200)

    results = m.evaluate(input_fn=test_input_fn, steps=1)
    for key in sorted(results):
        print "{0}, result: {1}".format(key, results[key])



