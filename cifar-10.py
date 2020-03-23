import tensorflow as tf
import numpy as np
 
# Dataset layout
def unpickle(file):
    import pickle
    with open('./cifar-10-python/'+file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
 
# onehot the images in library
def onehot(labels):
    onehot_labels = np.zeros([len(labels), 10])
    onehot_labels[np.arange(len(labels)), labels] = 1
    return onehot_labels
 

# deserialization cifar-10 training datas and test datas
data1 = unpickle('data_batch_1')
data2 = unpickle('data_batch_2')
data3 = unpickle('data_batch_3')
data4 = unpickle('data_batch_4')
data5 = unpickle('data_batch_5')

test_data = unpickle('test_batch')
 
# ajust the datas to 10000 pcs,3 channels, 32 lines, 32 columns
# then transpose 10000 pcs, 32 lines, 32 columns, 3 channels,

data1[b'data'] = data1[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).reshape(10000, -1)
data2[b'data'] = data2[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).reshape(10000, -1)
data3[b'data'] = data3[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).reshape(10000, -1)
data4[b'data'] = data4[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).reshape(10000, -1)
data5[b'data'] = data5[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).reshape(10000, -1)
test_data[b'data'] = test_data[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).reshape(10000, -1)
 
# combine 5 batches into 1
x_train = np.concatenate((data1[b'data'], data2[b'data'], data3[b'data'], data4[b'data'], data5[b'data']), axis=0)
y_train = np.concatenate((data1[b'labels'], data2[b'labels'], data3[b'labels'], data4[b'labels'], data5[b'labels']), axis=0)
y_train = onehot(y_train)
 
# obtain the training and test datas
x_test = test_data[b'data']
y_test = onehot(test_data[b'labels'])
 
# print shapes of datas
print('Training data shape:', x_train.shape)
print('Training labels shape:', y_train.shape)
print('Testing data shape:', x_test.shape)
print('Testing labels shape:', y_test.shape)
 
# parameters
lr = 1e-4  # learning rate
epoches = 10  # training times
batch_size = 500  # sample batch every time
n_batch = 20  # x_train.shape[0]//batch_size
n_features = 32*32*3
n_classes = 10
n_fc1 = 1024
n_fc2 = 512
 
# model
x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.float32, [None, n_classes])
 
# initialize the weights truncated normal
 
weight = {
    'conv1': tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.05)),
    'conv2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.05)),
    'fc1': tf.Variable(tf.truncated_normal([8*8*64, n_fc1], stddev=0.04)),
    'fc2': tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.04)),
    'fc3': tf.Variable(tf.truncated_normal([n_fc2, n_classes], stddev=1/512.0))
}
 
# initialize the bias 
 
bias = {
    'conv1': tf.Variable(tf.constant(0.1, shape=[32])),
    'conv2': tf.Variable(tf.constant(0.1, shape=[64])),
    'fc1': tf.Variable(tf.constant(0.1, shape=[n_fc1])),
    'fc2': tf.Variable(tf.constant(0.1, shape=[n_fc2])),
    'fc3': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}
 
x_input = tf.reshape(x, [-1, 32, 32, 3])

# visualize the input image
tf.summary.image('input', x_input, max_outputs=4)
 
 
conv1 = tf.nn.conv2d(x_input, weight['conv1'], [1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, bias['conv1'])
conv1 = tf.nn.relu(conv1)
 
img_conv1 = conv1[:, :, :, 0:1]
 
# visualize the feature in conv layer1
tf.summary.image('conv1', img_conv1, max_outputs=4)
 
pool1 = tf.nn.avg_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
 
#norm1 = tf.nn.lrn(pool1, depth_radius=4.0, bias=1.0, alpha=0.001/9.0, beta=0.75)
 
conv2 = tf.nn.conv2d(pool1, weight['conv2'], [1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, bias['conv2'])
conv2 = tf.nn.relu(conv2)
 
img_conv2 = conv2[:, :, :, 0:1]
 
# visualize the feature in conv layer2
tf.summary.image('conv2', img_conv2, max_outputs=4)
 
#norm2 = tf.nn.lrn(conv2, depth_radius=4.0, bias=1.0, alpha=0.001/9.0, beta=0.75)
 
pool2 = tf.nn.avg_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
 
reshape = tf.reshape(pool2, [-1, 8*8*64])
 
fc1 = tf.matmul(reshape, weight['fc1'])
fc1 = tf.nn.bias_add(fc1, bias['fc1'])
fc1 = tf.nn.relu(fc1)
 
fc2 = tf.matmul(fc1, weight['fc2'])
fc2 = tf.nn.bias_add(fc2, bias['fc2'])
fc2 = tf.nn.relu(fc2)
 
fc3 = tf.matmul(fc2, weight['fc3'])
fc3 = tf.nn.bias_add(fc3, bias['fc3'])
prediction = tf.nn.softmax(fc3)
 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
 
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
 
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(r'.\cifar-10\logs', sess.graph)
    merged = tf.summary.merge_all()
    for step in range(epoches):
        for batch in range(n_batch):
            # print(batch*batch_size, (batch + 1)*batch_size)
            xs = x_train[batch*batch_size: (batch + 1)*batch_size, :]
            ys = y_train[batch*batch_size: (batch + 1)*batch_size, :]
            summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y: ys})
            writer.add_summary(summary, batch)
            print(batch)
        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict={x: x_test[:2000, :], y: y_test[:2000, :]})
            print(step,acc)
