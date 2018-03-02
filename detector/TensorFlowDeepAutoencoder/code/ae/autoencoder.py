from __future__ import division
from __future__ import print_function
import time
from os.path import join as pjoin

import numpy as np
import tensorflow as tf
from detector.TensorFlowDeepAutoencoder.code.ae.utils.data import fill_feed_dict_ae, read_data_sets_pretraining
from detector.TensorFlowDeepAutoencoder.code.ae.utils.data import read_data_sets, fill_feed_dict
from detector.TensorFlowDeepAutoencoder.code.ae.utils.flags import FLAGS
from detector.TensorFlowDeepAutoencoder.code.ae.utils.eval import loss_supervised, evaluation, do_eval_summary,do_eval
from detector.TensorFlowDeepAutoencoder.code.ae.utils.utils import tile_raster_images
from time import gmtime, strftime
import os
class AutoEncoder(object):
  """Generic deep autoencoder.

  Autoencoder used for full training cycle, including
  unsupervised pretraining layers and final fine tuning.
  The user specifies the structure of the neural net
  by specifying number of inputs, the number of hidden
  units for each layer and the number of final output
  logits.
  """
  _weights_str = "weights{0}"
  _biases_str = "biases{0}"

  def __init__(self, shape, sess,restore_mode=None):
    """Autoencoder initializer

    Args:
      shape: list of ints specifying
              num input, hidden1 units,...hidden_n units, num logits
      sess: tensorflow session object to use
    """
    self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]
    self.__num_hidden_layers = len(self.__shape) - 2

    self.__variables = {}
    self.__sess = sess
    
    if(restore_mode) :
      self._restore_variables()
    else :
      self._setup_variables()


  @property
  def shape(self):
    return self.__shape

  @property
  def num_hidden_layers(self):
    return self.__num_hidden_layers

  @property
  def session(self):
    return self.__sess

  def __getitem__(self, item):
    """Get autoencoder tf variable

    Returns the specified variable created by this object.
    Names are weights#, biases#, biases#_out, weights#_fixed,
    biases#_fixed.

    Args:
     item: string, variables internal name
    Returns:
     Tensorflow variable
    """
    return self.__variables[item]

  def __setitem__(self, key, value):
    """Store a tensorflow variable

    NOTE: Don't call this explicity. It should
    be used only internally when setting up
    variables.

    Args:
      key: string, name of variable
      value: tensorflow variable
    """
    self.__variables[key] = value
  def _restore_variables(self):
   # print(tf.get_collection(tf.GraphKeys.VARIABLES, scope='autoencoder_variables/weights1'))

    #with tf.variable_scope("autoencoder_variables") as scope:
      # print(tf.get_collection(tf.GraphKeys.VARIABLES, scope="autoencoder_variables"))
      #print(scope)
      #tf.Variable 'autoencoder_variables/weights1:0' shape=(784, 2000)

 

      for i in range(self.__num_hidden_layers + 1):
        # Train weights
        name_w = self._weights_str.format(i + 1)
        w_shape = (self.__shape[i], self.__shape[i + 1])

        self[name_w] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder_variables/'+name_w)[0]

        # Train biases
        name_b = self._biases_str.format(i + 1)
        b_shape = (self.__shape[i + 1],)
        self[name_b] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder_variables/'+name_b)[0]

        if i < self.__num_hidden_layers:
          # Hidden layer fixed weights (after pretraining before fine tuning)
          self[name_w + "_fixed"] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder_variables/'+name_w + "_fixed")[0]
          # Hidden layer fixed biases
          self[name_b + "_fixed"] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder_variables/'+name_b + "_fixed")[0]
          # Pretraining output training biases
          name_b_out = self._biases_str.format(i + 1) + "_out"
          b_shape = (self.__shape[i],)

          self[name_b_out] =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='autoencoder_variables/'+name_b_out )[0]




  def _setup_variables(self):
    with tf.variable_scope("autoencoder_variables", reuse=True):
      for i in range(self.__num_hidden_layers + 1):
        # Train weights
        name_w = self._weights_str.format(i + 1)
        w_shape = (self.__shape[i], self.__shape[i + 1])
        a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
        w_init = tf.random_uniform(w_shape, -1 * a, a)
        self[name_w] = tf.Variable(w_init,
                                   name=name_w,
                                   trainable=True)
        # Train biases
        name_b = self._biases_str.format(i + 1)
        b_shape = (self.__shape[i + 1],)
        b_init = tf.zeros(b_shape)
        self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

        if i < self.__num_hidden_layers:
          # Hidden layer fixed weights (after pretraining before fine tuning)
          self[name_w + "_fixed"] = tf.Variable(tf.identity(self[name_w]),
                                                name=name_w + "_fixed",
                                                trainable=False)

          # Hidden layer fixed biases
          self[name_b + "_fixed"] = tf.Variable(tf.identity(self[name_b]),
                                                name=name_b + "_fixed",
                                                trainable=False)

          # Pretraining output training biases
          name_b_out = self._biases_str.format(i + 1) + "_out"
          b_shape = (self.__shape[i],)
          b_init = tf.zeros(b_shape)
          self[name_b_out] = tf.Variable(b_init,
                                         trainable=True,
                                         name=name_b_out)


  def _w(self, n, suffix=""):
    return self[self._weights_str.format(n) + suffix]

  def _b(self, n, suffix=""):
    return self[self._biases_str.format(n) + suffix]

  def get_variables_to_init(self, n):
    """Return variables that need initialization

    This method aides in the initialization of variables
    before training begins at step n. The returned
    list should be than used as the input to
    tf.initialize_variables

    Args:
      n: int giving step of training
    """
    assert n > 0
    assert n <= self.__num_hidden_layers + 1

    vars_to_init = [self._w(n), self._b(n)]

    if n <= self.__num_hidden_layers:
      vars_to_init.append(self._b(n, "_out"))

    if 1 < n <= self.__num_hidden_layers:
      vars_to_init.append(self._w(n - 1, "_fixed"))
      vars_to_init.append(self._b(n - 1, "_fixed"))

    return vars_to_init

  @staticmethod
  def _activate(x, w, b, transpose_w=False):

    y = tf.sigmoid(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
    return y

  def pretrain_net(self, input_pl, n, is_target=False):
    """Return net for step n training or target net

    Args:
      input_pl:  tensorflow placeholder of AE inputs
      n:         int specifying pretrain step
      is_target: bool specifying if required tensor
                  should be the target tensor
    Returns:
      Tensor giving pretraining net or pretraining target
    """
    assert n > 0
    assert n <= self.__num_hidden_layers

    last_output = input_pl
    for i in range(n - 1):
      w = self._w(i + 1, "_fixed")
      b = self._b(i + 1, "_fixed")

      last_output = self._activate(last_output, w, b)

    if is_target:
      return last_output

    last_output = self._activate(last_output, self._w(n), self._b(n))

    out = self._activate(last_output, self._w(n), self._b(n, "_out"),
                         transpose_w=True)
    out = tf.maximum(out, 1.e-9)
    out = tf.minimum(out, 1 - 1.e-9)
    return out

  def supervised_net(self, input_pl):
    """Get the supervised fine tuning net

    Args:
      input_pl: tf placeholder for ae input data
    Returns:
      Tensor giving full ae net
    """
    last_output = input_pl

    for i in range(self.__num_hidden_layers + 1):
      # Fine tuning will be done on these variables

      w = self._w(i + 1)
      b = self._b(i + 1)


      last_output = self._activate(last_output, w, b)

    return last_output


loss_summaries = {}



def training(loss, learning_rate, loss_key=None):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    loss_key: int giving stage of pretraining so we can store
                loss summaries for each pretraining stage

  Returns:
    train_op: The Op for training.
  """
  if loss_key is not None:
    # Add a scalar summary for the snapshot loss.
    loss_summaries[loss_key] = tf.summary.scalar(loss.op.name, loss)
  else:
    tf.summary.scalar(loss.op.name, loss)
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op, global_step


def loss_x_entropy(output, target):
  """Cross entropy loss

  See https://en.wikipedia.org/wiki/Cross_entropy

  Args:
    output: tensor of net output
    target: tensor of net we are trying to reconstruct
  Returns:
    Scalar tensor of cross entropy
  """
  with tf.name_scope("xentropy_loss"):
      net_output_tf = tf.convert_to_tensor(output, name='input')
      target_tf = tf.convert_to_tensor(target, name='target')
      # min_net = tf.abs(tf.reduce_min(net_output_tf))+1
      min_net = 1e-10
      cross_entropy = tf.add(tf.multiply(tf.log(net_output_tf+min_net, name='log_output'),
                                    target_tf),
                             tf.multiply(tf.log(1 - net_output_tf+min_net),
                                    (1 - target_tf)))

      # p=0.01
      # penalTerm = tf.add(tf.multiply(p,tf.log(tf.div(p,net_output_tf), name='log_output')),
      # tf.multiply(1-p,   tf.log( tf.div(1-p,1-target_tf ) ) )    )
      # cross_entropy=tf.add(tf.losses.mean_squared_error(net_output_tf,target_tf),2*penalTerm )
      return  -1*tf.reduce_mean(tf.reduce_sum(cross_entropy, 1),
                                 name='xentropy_mean')
def plot_filter(sess,ae,summary_writer,i,fine_tune) :
        import math as mt
        a = i+1
        size = int( mt.sqrt(int(ae["weights"+str(a)].shape[0])) )

        filters = sess.run(tf.identity(ae["weights"+str(a)]) )
        np.save(pjoin(FLAGS.chkpt_dir, fine_tune+"filters"+str(a)), filters)
        filters = tile_raster_images(X=filters.T,
                                     img_shape=(size,
                                                size),
                                     tile_shape=(10, 10),
                                     output_pixel_vals=False)
        filters = np.expand_dims(np.expand_dims(filters, 0), 3)
        image_var = tf.Variable(filters)
        image_filter = tf.identity(image_var)
        sess.run(tf.initialize_variables([image_var]))
        img_filter_summary_op = tf.summary.image(fine_tune+str(a)+"_layer_filters",
                                                 image_filter)
        summary_writer.add_summary(sess.run(img_filter_summary_op))
        summary_writer.flush()

def main_unsupervised():

  with tf.Graph().as_default() as g:
    var = []
    

    #config = tf.ConfigProto(log_device_placement=True)
    #config.gpu_options.allow_growth = True
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    

    num_hidden = FLAGS.num_hidden_layers
    ae_hidden_shapes = [getattr(FLAGS, "hidden{0}_units".format(j + 1))
                        for j in range(num_hidden)]
    ae_shape = [FLAGS.image_pixels] + ae_hidden_shapes + [FLAGS.num_classes]

    ae = AutoEncoder(ae_shape, sess)

    data = read_data_sets_pretraining(FLAGS.data_dir)

    num_train = data.train.num_examples

    learning_rates = {j: getattr(FLAGS,
                                 "pre_layer{0}_learning_rate".format(j + 1))
                      for j in range(num_hidden)}

    noise = {j: getattr(FLAGS, "noise_{0}".format(j + 1))
             for j in range(num_hidden)}

    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph_def)

    for i in range(len(ae_shape) - 2):
      n = i + 1

      with tf.variable_scope("pretrain_{0}".format(n)):
        
        print(ae_shape[0])
        input_ = tf.placeholder(dtype=tf.float32,
                                shape=(FLAGS.batch_size, ae_shape[0]),
                                name='ae_input_pl')
        target_ = tf.placeholder(dtype=tf.float32,
                                 shape=(FLAGS.batch_size, ae_shape[0]),
                                 name='ae_target_pl')
        layer = ae.pretrain_net(input_, n)

        with tf.name_scope("target"):
          target_for_loss = ae.pretrain_net(target_, n, is_target=True)

        loss = loss_x_entropy(layer, target_for_loss)

        train_op, global_step = training(loss, learning_rates[i], i)

        summary_dir = pjoin(FLAGS.summary_dir, 'pretraining_{0}'.format(n))

        # summary_writer = tf.train.SummaryWriter(summary_dir,
        #                                         graph_def=sess.graph_def,
        #                                         flush_secs=FLAGS.flush_secs)

        summary_vars = [ae["biases{0}".format(n)], ae["weights{0}".format(n)]]

        hist_summarries = [tf.summary.histogram(v.op.name, v)
                           for v in summary_vars]
        hist_summarries.append(loss_summaries[i])

        summary_op = tf.summary.merge(hist_summarries)
 
        vars_to_init = ae.get_variables_to_init(n)
        vars_to_init.append(global_step)

        sess.run(tf.initialize_variables(vars_to_init))



        print("\n\n")
        print("| Training Step | Cross Entropy |  Layer  |   Epoch  |")
        print("|---------------|---------------|---------|----------|")
        steps = (num_train//FLAGS.batch_size)
        epoch=FLAGS.pretraining_epochs
        if(n>1) :
          epoch=50
        for k in range(epoch):
          for step in range(steps ):


            feed_dict = fill_feed_dict_ae(data.train, input_, target_, noise[i])
 
            loss_summary, loss_value = sess.run([train_op, loss],
                                                feed_dict=feed_dict)

            if step % 1 == 0:
              summary_str = sess.run(summary_op, feed_dict=feed_dict)
              summary_writer.add_summary(summary_str, step)

              # image_op = \
              #     tf.summary.image("training_images",
              #                      tf.reshape(input_,
              #                                 (FLAGS.batch_size,
              #                                  FLAGS.image_size,
              #                                  FLAGS.image_size, 1)),
              #                      max_outputs=10)

              # summary_img_str = sess.run(image_op,
              #                            feed_dict=feed_dict)
              # summary_writer.add_summary(summary_img_str)

              output = "| {0:>13}/{1} | {2:13.4f} | Layer {3} | Epoch {4}  |"\
                       .format(step,steps,loss_value, n, k)
                        #step//num_train + 1)

              print(output)
      if i == 0:

        plot_filter(sess,ae,summary_writer,i,"")

    # folder = "model_unsps_"+str(strftime("%Y-%m-%d_%H:%M:%S", gmtime()))
    # os.mkdir(folder)
    # folder += "/model"
    # saver.save(sess, folder)
  return ae

def save_model(sess) :
    from time import gmtime, strftime
    import os
    saver = tf.train.Saver()
    folder = "model_sps_"+str(strftime("%Y-%m-%d-%H:%M:%S", gmtime()))
    os.mkdir(folder)
    folder += "/model"
    saver.save(sess, folder)
def main_supervised(ae):

  with ae.session.graph.as_default():

    sess = ae.session
    input_pl = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,
                                                 FLAGS.image_pixels),
                              name='input_pl')
    logits = ae.supervised_net(input_pl)

    data = read_data_sets(FLAGS.data_dir)
    num_train = data.train.num_examples

    labels_placeholder = tf.placeholder(tf.int32,
                                        shape=FLAGS.batch_size,
                                        name='target_pl')

    loss = loss_supervised(logits, labels_placeholder)
    train_op, global_step = training(loss, FLAGS.supervised_learning_rate)
    eval_correct = evaluation(logits, labels_placeholder)

    hist_summaries = [ae['biases{0}'.format(i + 1)]
                      for i in range(ae.num_hidden_layers + 1)]
    hist_summaries.extend([ae['weights{0}'.format(i + 1)]
                           for i in range(ae.num_hidden_layers + 1)])

    hist_summaries = [tf.summary.histogram(v.op.name + "_fine_tuning", v)
                      for v in hist_summaries]
    summary_op = tf.summary.merge(hist_summaries)

    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph_def)
    # tf.train.SummaryWriter(pjoin(FLAGS.summary_dir,
    #                                               'fine_tuning'),
    #                                         graph_def=sess.graph_def,
    #                                         flush_secs=FLAGS.flush_secs)

    vars_to_init = ae.get_variables_to_init(ae.num_hidden_layers + 1)
    vars_to_init.append(global_step)
    #sess.run(tf.initialize_variables(vars_to_init))
    init = tf.initialize_all_variables() 
    sess.run(init)

    steps =  (num_train//FLAGS.batch_size)
    for k in range(FLAGS.finetuning_epochs):
      for step in range(steps):
        start_time = time.time()

        feed_dict = fill_feed_dict(data.train,
                                   input_pl,
                                   labels_placeholder)

        _, loss_value = sess.run([train_op, loss],
                                 feed_dict=feed_dict)

        duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        if step % 1 == 0:
          # Print status to stdout.
          print('Step %d/%d: loss = %.2f (%.3f sec) epoch= %d ' % (step, steps,loss_value, duration,k))
          # Update the events file.

          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)
          # summary_img_str = sess.run(
          #     tf.summary.image("training_images",
          #                      tf.reshape(input_pl,
          #                                 (FLAGS.batch_size,
          #                                  FLAGS.image_size,
          #                                  FLAGS.image_size, 1)),
          #                      max_outputs=10),
          #     feed_dict=feed_dict
          # )
          # summary_writer.add_summary(summary_img_str)

        if (step + 1) % 1000 == 0 or (step + 1) == steps:
          train_sum = do_eval_summary("training_error",
                                      sess,
                                      eval_correct,
                                      input_pl,
                                      labels_placeholder,
                                      data.train)

          # val_sum = do_eval_summary("validation_error",
          #                           sess,
          #                           eval_correct,
          #                           input_pl,
          #                           labels_placeholder,
          #                           data.validation)

          # test_sum = do_eval_summary("test_error",
          #                            sess,
          #                            eval_correct,
          #                            input_pl,
          #                            labels_placeholder,
          #                            data.test)

          # summary_writer.add_summary(train_sum, step)
          # summary_writer.add_summary(val_sum, step)
          # summary_writer.add_summary(test_sum, step)

    num_hidden = FLAGS.num_hidden_layers
    ae_hidden_shapes = [getattr(FLAGS, "hidden{0}_units".format(j + 1))
                        for j in range(num_hidden)]
    ae_shape = [FLAGS.image_pixels] + ae_hidden_shapes + [FLAGS.num_classes]
    #for i in range(len(ae_shape) - 2):
        #plot_filter(sess,ae,summary_writer,i,"fine_tuned_")
    save_model(sess)
    # do_eval(sess,
    #     eval_correct,
    #     input_pl,
    #     labels_placeholder,
    #     data.test)
    # from ae import autoencoder_test as autoencoder_test
    # test = autoencoder_test.AutoEncoderTest()
    # test.test_nets(ae, data.test)



if __name__ == '__main__':
  ae = main_unsupervised()
  main_supervised(ae)