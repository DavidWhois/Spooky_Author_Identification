
# coding: utf-8

import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_path = 'train.csv'
data = pd.read_csv(data_path)
data.head()

data_path = 'test.csv'
test_data = pd.read_csv(data_path)
test_data.head()

'''
def clean_stop_words(words,stopwords=stopwords):
    cleaned_words = []
    for word in words:
        if word not in stopwords:
            cleaned_words.append(word)
    return cleaned_words
'''
from collections import Counter
counts = Counter()
for line in data['text']:
    words = line.lower().split()
    #words = clean_stop_words(words)
    counts.update(words)

for line in test_data['text']:
    words = line.lower().split()
    #words = clean_stop_words(words)
    counts.update(words)

vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 0)}

def text_to_ints(text):
    text_ints = [vocab_to_int[word] for word in text.lower().split()]
    text_ints = np.array(text_ints)
    return text_ints

train_X = []
text_length = []
for each in data['text']:
    train_X.append(text_to_ints(each))
    text_length.append(len(text_to_ints(each)))
text_length = np.array(text_length)
train_X = np.array(train_X)

labels = []
for each in data['author']:
    if each=='EAP':
        labels.append(0)
    elif each=='HPL':
        labels.append(1)
    else:
        labels.append(2)

int_to_author={0:'EAP',1:'HPL',2:'MWS'}
train_y = np.array(labels)
train_y = pd.get_dummies(train_y)

def trunc_seq(seq,seq_len=100):
    features = np.zeros((len(seq), seq_len), dtype=int)
    for i, row in enumerate(seq):
        features[i, -len(row):] = np.array(row)[:seq_len]
    return features
train_X = trunc_seq(train_X)

#############超参调试区###################

lstm_size = 256
lstm_layers = 1
batch_size = 100
learning_rate = 0.001
epochs = 8

############超参调试区####################
n_words = len(vocab_to_int)
graph = tf.Graph()

with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

embed_size = 300

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)

with graph.as_default():
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed,initial_state=initial_state)
with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 3, activation_fn=tf.nn.softmax)
    #cost = tf.losses.mean_squared_error(labels_, predictions)
    cost = tf.losses.softmax_cross_entropy(labels_,predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)

def get_batches(x, y, batch_size=100):

    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)

        for ii, (x, y) in enumerate(get_batches(train_X, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y,
                    keep_prob: 0.6,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

            if iteration%5==0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))
            iteration +=1
    saver.save(sess, "checkpoints/sentiment.ckpt")

test_X = []
test_text_length = []
for each in test_data['text']:
    test_X.append(text_to_ints(each))
    test_text_length.append(len(text_to_ints(each)))
test_text_length = np.array(text_length)

import numpy as np
zeros = np.array([0,0,0])
for i in range(9000-len(test_X)):
    test_X.append(zeros)

test_X = np.array(test_X)
test_X = trunc_seq(test_X)

def test_get_batches(x, batch_size=100):
    n_batches = len(x)//batch_size
    x = x[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size]

n_batches = len(test_X)//batch_size
state_size = len(test_X)-n_batches*batch_size

predicts = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, x in enumerate(test_get_batches(test_X, batch_size), 1):
        feed = {inputs_: x,
                keep_prob: 1,
                initial_state: test_state}
        predict = sess.run(predictions, feed_dict=feed)
        predicts.append(predict)

result = []
for each in predicts:
    for i in range(each.shape[0]):
        prob = each[i,:]
        result.append(prob)

result_array = np.array(result)
result_df = pd.DataFrame(result_array,dtype=float)
result_df.columns=['EAP','HPL','MWS']
result_df = result_df[:len(test_data['id'])]
result_df = result_df.applymap(lambda x: '%.15f' % x)
result_df.insert(0,'id',test_data['id'])
result_df.head()

result_df.to_csv('submision.csv',index=False)
