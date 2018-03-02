import tensorflow as tf

def weight_variable(shape, varname):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="W_"+varname)

def bias_variable(shape, varname):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="b_"+varname)

def conv2d(x, W, stride, padd):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padd)

def maxpool(h, kernel_sz, stride, padd):
    return tf.nn.max_pool(h, ksize=[1, kernel_sz, kernel_sz, 1], stride=[1, stride, stride, 1], padding=padd)

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)

        return f1 * x + f2 * abs(x)

x = tf.placeholder(tf.float32, shape=[None, 96, 96, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 5])
keep_prob = tf.placeholder(tf.float32)

W = weight_variable([3, 3, 3, 8], "conv1")
# tf.histogram_summary("W_conv1", W)
b = bias_variable([8], 'conv1')
conv = conv2d(x, W, 1, 'SAME')
h = lrelu(tf.nn.bias_add(conv, b))
net = maxpool(h, 2, 2, 'SAME')

W = weight_variable([3, 3, 8, 16], "conv2")
# tf.histogram_summary("W_conv2", W)
b = bias_variable([16], 'conv2')
conv = conv2d(net, W, 1, 'SAME')
h = lrelu(tf.nn.bias_add(conv, b))
net = maxpool(h, 2, 2, 'SAME')

W = weight_variable([1, 1, 16, 8], "conv3")
# tf.histogram_summary("W_conv3", W)
b = bias_variable([8], 'conv3')
conv = conv2d(net, W, 1, 'SAME')
net = lrelu(tf.nn.bias_add(conv, b))

W = weight_variable([3, 3, 8, 64], "conv4")
#tf.histogram_summary("W_conv4", W)
b = bias_variable([64], "conv4")
conv = conv2d(net, W, 1, 'SAME')
net = lrelu(tf.nn.bias_add(conv, b))

W = weight_variable([1, 1, 64, 8], "conv5")
#tf.histogram_summary("W_conv5", W)
b = bias_variable([8], "conv5")
conv = conv2d(net, W, 1, 'SAME')
net = lrelu(tf.nn.bias_add(conv, b))

W = weight_variable([3, 3, 8, 64], "conv6")
#tf.histogram_summary("W_conv6", W)
b = bias_variable([64], "conv6")
conv = conv2d(net, W, 1, 'SAME')
h = lrelu(tf.nn.bias_add(conv, b))
net = maxpool(h, 2, 2, 'SAME')

W = weight_variable([1, 1, 64, 16], "conv7")
#tf.histogram_summary("W_conv7", W)
b = bias_variable([16], "conv7")
conv = conv2d(net, W, 1, 'SAME')
net = lrelu(tf.nn.bias_add(conv, b))

W = weight_variable([3, 3, 16, 128], "conv8")
#tf.histogram_summary("W_conv8", W)
b = bias_variable([128], "conv8")
conv = conv2d(net, W, 1, 'SAME')
net = lrelu(tf.nn.bias_add(conv, b))

W = weight_variable([1, 1, 128, 16], "conv9")
#tf.histogram_summary("W_conv9", W)
b = bias_variable([16], "conv9")
conv = conv2d(net, W, 1, 'SAME')
net = lrelu(tf.nn.bias_add(conv, b))

W = weight_variable([3, 3, 16, 128], "conv10")
#tf.histogram_summary("W_conv10", W)
b = bias_variable([128], "conv10")
conv = conv2d(net, W, 1, 'SAME')
h = lrelu(tf.nn.bias_add(conv, b))
net = maxpool(h, 2, 2, 'SAME')

W = weight_variable([1, 1, 128, 32], "conv11")
#tf.histogram_summary("W_conv11", W)
b = bias_variable([32], "conv11")
conv = conv2d(net, W, 1, 'SAME')
net = lrelu(tf.nn.bias_add(conv, b))

W = weight_variable([3, 3, 32, 256], "conv12")
#tf.histogram_summary("W_conv12", W)
b = bias_variable([256], "conv12")
conv = conv2d(net, W, 1, 'SAME')
net = lrelu(tf.nn.bias_add(conv, b))

W = weight_variable([1, 1, 256, 32], "conv13")
#tf.histogram_summary("W_conv13", W)
b = bias_variable([32], "conv13")
conv = conv2d(net, W, 1, 'SAME')
net = lrelu(tf.nn.bias_add(conv, b))

W = weight_variable([3, 3, 32, 256], "conv14")
#tf.histogram_summary("W_conv14", W)
b = bias_variable([256], "conv14")
conv = conv2d(net, W, 1, 'SAME')
net = lrelu(tf.nn.bias_add(conv, b))

W = weight_variable([1, 1, 256, 64], "conv15")
#tf.histogram_summary("W_conv15", W)
b = bias_variable([64], "conv15")
conv = conv2d(net, W, 1, 'SAME')
net = lrelu(tf.nn.bias_add(conv, b))


W = weight_variable([1, 1, 64, 5], "conv16")
#tf.histogram_summary("W_conv16", W)
b = bias_variable([5], "conv16")
conv = conv2d(net, W, 1, 'SAME')
net = tf.nn.bias_add(conv, b)
print(net.get_shape())

net = tf.nn.avg_pool(net, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')
print(net.get_shape())
logits = tf.reshape(net, [-1, 5]) #5 is number of class
