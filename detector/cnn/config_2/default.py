
import tensorflow as tf

# Configuration
n_classes = 2
epochs = 50
batch_size = 16
image_size = 32
h_image_size = 10
h_image_size = 10
dropout = 0.75

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Store layers weight & bias
weights = {
           # 5x5, 3 input (channel), 32 outputs
           'wc1' : weight_variable([25,25,1,96]),#tf.Variable(tf.random_normal([5,5,3,32])),

           # 5x5 32 input (channel), 64 outputs
           'wc2' : weight_variable([5,5,96,256]),#tf.Variable(tf.random_normal([5,5,32,64])),


           'wc3' : weight_variable([4,4,96,256]),

           # fully connected
           #'wd1' : weight_variable([4*4*256, 1024]),#tf.Variable(tf.random_normal([6*6*64, 1024])),
           'wd1' : weight_variable([3*3*256, 1024]),
           # 1024 inputs, 10 outpus (class prediction)
           'out' : weight_variable([1024, n_classes])#tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
          'bc1' : bias_variable([96]),#tf.Variable(tf.random_normal([32])),
          'bc2' : bias_variable([256]),#tf.Variable(tf.random_normal([64])),
          'bc3' : bias_variable([256]),
          'bd1' : bias_variable([1024]),#tf.Variable(tf.random_normal([1024])),
          'out' : bias_variable([n_classes])#tf.Variable(tf.random_normal([n_classes]))
}



