import pandas as pd
import numpy as np
import tensorflow as tf



def inference(data, k):

    with tf.name_scope('interaction'):
        u = tf.Variable(tf.random_normal([data.shape[0], k]), name='u')
        v = tf.Variable(tf.random_normal([k, data.shape[1]]), name='v')

        interaction = tf.matmul(u,v)

    return interaction

def loss(pred_mat, labels):
    

    loss = tf.sparse_reduce_sum(tf.sparse_add(labels, -1.0*pred_mat)**2)
    return loss


def training(loss, learning_rate):


    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optmizer.minimize(loss, global_step=global_step)
    return train_op

def evaluate(pred_mat, labels):
    return tf.reduce_mean(pred_mat-labels)**2)



if __name__ == '__main__':
    df = pd.read_csv('books.csv')
    
    #users = tf.contrib
