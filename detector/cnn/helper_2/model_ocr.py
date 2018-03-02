from detector.cnn.helper_2 import model_base
import detector.cnn.config_2.default as config
import tensorflow as tf


x = tf.placeholder(tf.float32, shape=[None,config.image_size,config.image_size,1])

#x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
#x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_ = tf.placeholder(tf.float32, shape=[None, config.n_classes])




keep_prob = tf.placeholder(tf.float32)

# Convolutional layer
print( x.get_shape())
conv1 = model_base.conv2d(x, config.weights['wc1'], config.biases['bc1'], 1, 'conv1', True)

# Max Pooling (down-sampling)
print( conv1.get_shape() )
conv1 = model_base.maxpool2d(conv1, k=7)
print( conv1.get_shape() )
# Convolutional layer
conv2 = model_base.conv2d(conv1, config.weights['wc2'], config.biases['bc2'], 1, 'conv2', True)
# Max Pooling (down-sampling)
print( conv2.get_shape())
conv2 = model_base.maxpool2d(conv2, k=2)
print( conv2.get_shape())


# conv3 = model_base.conv2d(conv2, config.weights['wc3'], config.biases['bc3'], 1, 'conv3', True)
# print( conv3.get_shape())
# conv3 = model_base.maxpool2d(conv3, k=2)
# print( conv3.get_shape())
# Fully connected layer
# Reshape conv2 output to fit fully connected layer input
print(  config.weights['wd1'].get_shape().as_list()[0])
#fc1 = tf.reshape(conv3, [-1, config.weights['wd1'].get_shape().as_list()[0]])

fc1 = tf.reshape(conv2, [-1, config.weights['wd1'].get_shape().as_list()[0]] )



fc1 = tf.add(tf.matmul(fc1, config.weights['wd1']), config.biases['bd1'])
fc1 = tf.nn.relu(fc1)

# Apply Dropout
fc1 = tf.nn.dropout(fc1, config.dropout)

logits = tf.add(tf.matmul(fc1, config.weights['out']), config.biases['out'])

