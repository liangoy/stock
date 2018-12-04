import sys

sys.path.append('.')

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from config import ROOT_PATH

dtype = tf.float32
long = 20
batch_size = 512
otype = 1

data_bp = pd.read_csv(ROOT_PATH + '/data/bp.csv').dropna()
data_bp = data_bp.drop(['Adj Close', 'Volume'], axis=1)
data_hs = pd.read_csv(ROOT_PATH + '/data/hs.csv').dropna()
data_hs = data_hs.drop(['Adj Close', 'Volume'], axis=1)
data_jp = pd.read_csv(ROOT_PATH + '/data/jp.csv').dropna()
data_jp = data_jp.drop(['Adj Close', 'Volume'], axis=1)
data_vix = pd.read_csv(ROOT_PATH + '/data/vix.csv').dropna()
data_vix = data_vix.drop(['Adj Close', 'Volume'], axis=1)

data = pd.merge(data_bp, data_hs, on='Date', how='left')
data = pd.merge(data, data_jp, on='Date', how='left')
data = pd.merge(data, data_vix, on='Date', how='left').sort_values(by='Date')
data = data.drop('Date', axis=1)
data = data.iloc[300:].replace(0, None)
data = data.fillna(method='ffill')

data = np.array(data)[1:]
data = np.array(data)
data_t = data[1:]
data_t_1 = data[:-1] + 0.0000001

'''['Open_x', 'High_x', 'Low_x', 'Close_x', 'Open_y', 'High_y', 'Low_y','Close_y', 'Open_x', 'High_x', 'Low_x', 'Close_x', 'Open_y', 'High_y','Low_y', 'Close_y']'''
for i in range(15):
    data_t[:, i] /= data_t_1[:, i // 4 * 4 + 3]
data = data_t - 1

'''
标准化
'''
for i in range(4):
    data[:, i * 4:i * 4 + 4] = (data[:, i * 4:i * 4 + 4] - data[:, i * 4:i * 4 + 4].mean()) / data[:,
                                                                                              i * 4:i * 4 + 4].std()

data_x, data_y = [], []
for i in range(len(data) - long):
    data_x.append(data[i:i + long])
    data_y.append(data[i + long, otype] - data[i + long, 0])

data_x = np.array(data_x)
data_y = np.array(data_y)
data_train_x, data_train_y = data_x[:-batch_size], data_y[:-batch_size]
data_test_x, data_test_y = data_x[-batch_size:], data_y[-batch_size:]

train_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.constant(data_train_x, dtype=dtype), tf.constant(data_train_y, dtype=dtype))).repeat().batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.constant(data_test_x, dtype=dtype), tf.constant(data_test_y, dtype=dtype))).repeat().batch(batch_size)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, output_types=test_dataset.output_types,
                                               output_shapes=test_dataset.output_shapes)
next_element = iterator.get_next()
train_iterator = train_dataset.make_one_shot_iterator()
test_iterator = test_dataset.make_initializable_iterator()

x, y_ = iterator.get_next()

x = tf.reshape(x, shape=[batch_size, x.shape[1], x.shape[2]])
