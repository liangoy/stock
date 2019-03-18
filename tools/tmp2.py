import sys

sys.path.append('.')

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from config import ROOT_PATH
import pymongo
import sys

client = pymongo.MongoClient()
col = client['stock']['highlow2']
name = sys.argv[1]  # eg "vix_vix_vix_vix_vix"

dtype = tf.float32
long = 20
batch_size = 512
otype = 1

data_dict = {i: pd.read_csv(ROOT_PATH + '/data/' + j + '.csv').dropna() for i, j in enumerate(name.split('_'))}

'''
前复权
'''
for i in data_dict:
    adj = data_dict[i]['Adj Close'] / data_dict[i]['Close']
    data_dict[i]['Open'] *= adj
    data_dict[i]['High'] *= adj
    data_dict[i]['Low'] *= adj
    data_dict[i]['Close'] *= adj
    data_dict[i].drop(['Adj Close', 'Volume'], axis=1, inplace=True)

data = data_dict[0]
for i in range(1, len(data_dict)):
    data = pd.merge(data, data_dict[i], on='Date', how='left')

data = data.sort_values(by='Date')
data = data.drop('Date', axis=1)
data = data.replace(0, None)
data = data.fillna(method='ffill')

data = np.array(data)[1:]
data_t = data[1:]
data_t_1 = data[:-1] + 0.0000001

'''['Open_x', 'High_x', 'Low_x', 'Close_x', 'Open_y', 'High_y', 'Low_y','Close_y', 'Open_x', 'High_x', 'Low_x', 'Close_x', 'Open_y', 'High_y','Low_y', 'Close_y']'''
for i in range(data.shape[1] - 1):
    data_t[:, i] /= data_t_1[:, i // 4 * 4 + 3]
data = data_t - 1

'''
标准化
'''
for i in range(len(data_dict)):
    data[:, i * 4:i * 4 + 4] = (data[:, i * 4:i * 4 + 4] - data[:, i * 4:i * 4 + 4].mean()) / data[:,
                                                                                              i * 4:i * 4 + 4].std()
'''
shuffle!!!!
'''
# np.random.shuffle(data)


data_x, data_y = [], []
for i in range(len(data) - long):
    data_x.append(data[i:i + long])
    data_y.append(data[i + long, 1] - data[i + long, 2])

data_x = np.array(data_x)
data_y = np.array(data_y)
data_y = (data_y - data_y.mean()) / data_y.std()
data_train_x, data_train_y = data_x[:-batch_size], data_y[:-batch_size]
data_test_x, data_test_y = data_x[-batch_size:], data_y[-batch_size:]

train_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.constant(data_train_x, dtype=dtype), tf.constant(data_train_y, dtype=dtype))).repeat().shuffle(10000).prefetch(
    100000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.constant(data_test_x, dtype=dtype), tf.constant(data_test_y, dtype=dtype))).repeat().batch(batch_size)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, output_types=test_dataset.output_types,
                                               output_shapes=test_dataset.output_shapes)
next_element = iterator.get_next()
train_iterator = train_dataset.make_one_shot_iterator()
test_iterator = test_dataset.make_initializable_iterator()

x, y_ = iterator.get_next()

X = tf.reshape(x, shape=[batch_size, x.shape[1], x.shape[2]])

# X = tf.layers.batch_normalization(x, training=True, scale=False, center=False, axis=[0, -1])

gru = tf.nn.rnn_cell.GRUCell(num_units=128)
state = gru.zero_state(batch_size, dtype=dtype)
with tf.variable_scope('RNN'):
    for timestep in range(long):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = gru(X[:, timestep], state)

out = tf.nn.relu(tf.layers.dense(cell_output, 8))

y = tf.layers.dense(out, 1)[:, 0]

loss = tf.reduce_mean((y - y_) * (y - y_))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
train_handle = sess.run(train_iterator.string_handle())
test_handle = sess.run(test_iterator.string_handle())
sess.run(tf.global_variables_initializer())
sess.run(test_iterator.initializer)

cnt = 0
corr = -2
for i in range(10 ** 10):
    sess.run(optimizer, feed_dict={handle: train_handle})
    if not i % 100:
        loss_train, y_train, y_train_ = sess.run([loss, y, y_], feed_dict={handle: train_handle})
        str_train = str(('train: ', loss_train, np.mean(np.abs(y_train - y_train_)) / np.mean(np.abs(y_train_)),
                         np.corrcoef(y_train, y_train_)[1, 0]))
        loss_test, y_test, y_test_ = sess.run([loss, y, y_], feed_dict={handle: test_handle})
        str_test = str(('test:  ', loss_test, np.mean(np.abs(y_test - y_test_)) / np.mean(np.abs(y_test_)),
                        np.corrcoef(y_test, y_test_)[1, 0]))

        '''
        keep
        '''
        if corr > np.corrcoef(y_test, y_test_)[1, 0]:
            cnt += 1
        else:
            cnt = 0
            corr = np.corrcoef(y_test, y_test_)[1, 0]
        if cnt == 30:
            col.insert_one({'name': name, 'corr': corr})
            break
        print(str_train, str_test, len([i for i in y_test if i > y_test[-2]]))
