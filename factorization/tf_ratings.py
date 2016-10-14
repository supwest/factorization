import pandas as pd
import numpy as np
import tensorflow as tf



def inference(data, k):

    with tf.name_scope('interaction'):
        #u = tf.Variable(tf.random_normal([70000, k]), name='u')
        #v = tf.Variable(tf.random_normal([k, 15]), name='v')
        b = tf.Variable(tf.random_normal([14,1]), name='b')
        interaction = tf.matmul(data, b)

    return interaction

def loss(pred_mat, labels):
    

    #loss = tf.sparse_reduce_sum(tf.sparse_add(labels, -1.0*pred_mat)**2)
    loss = tf.reduce_sum((labels-pred_mat)**2)
    return loss


def training(loss, learning_rate):


    tf.scalar_summary(loss.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    global_step = tf.Variable(0.000001, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluate(pred_mat, labels):
    return tf.reduce_mean((pred_mat-labels)**2)

def get_data():
    df = pd.read_csv('/home/cully/returnpath/interview_data.csv')
    df = df.drop(['from_domain_hash', 'Domain_extension', 'day', 'id'], axis=1)
    return df

def placeholder_inputs(batch_size):
    data_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 14))
    read_rate_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
    return data_placeholder, read_rate_placeholder

def fill_feed_dict(df, data_pl, read_rate_pl):
    df2 = df.copy()
    y = df2.pop('read_rate').values
    X = df2.values

    feed_dict = {data_pl: X[:1000], read_rate_pl: y[:1000]}
    return feed_dict

def do_eval(sess, eval_error, data_placeholder, read_rate_placeholder, df):
    err = []
    steps_per_epoch = df.shape[0] // 1000
    num_examples = steps_per_epoch * 1000
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(df, data_placeholder, read_rate_placeholder)
        err.append(sess.run(eval_error, feed_dict=feed_dict))

    print ("err: {}".format(np.mean(err)))

def run_training():
    df = get_data()
    with tf.Graph().as_default():
        data_placeholder, read_rate_placeholder = placeholder_inputs(1000)

        pred_rates = inference(data_placeholder, 5)

        l = loss(pred_rates, read_rate_placeholder)

        train_op = training(l, 0.000001)

        eval_error = evaluate(pred_rates, read_rate_placeholder)

        summary = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        saver = tf.train.Saver()

        sess = tf.Session()

        summary_writer = tf.train.SummaryWriter('/home/cully/git/factorization/factorization/')

        sess.run(init)

        for step in xrange(2):

            feed_dict = fill_feed_dict(df, data_placeholder, read_rate_placeholder)
            _, loss_value = sess.run([train_op, l], feed_dict=feed_dict)
    
            do_eval(sess, eval_error, data_placeholder, read_rate_placeholder, df)

        saver.save(sess, 'my_model')

def main(_):
    run_training()

if __name__ == '__main__':
    df = pd.read_csv('books.csv')
    tf.app.run()
    #users = tf.contrib
