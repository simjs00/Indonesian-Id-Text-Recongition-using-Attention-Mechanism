from __future__ import division
from __future__ import print_function

import tensorflow as tf

from detector.TensorFlowDeepAutoencoder.code.ae.autoencoder import  AutoEncoder
import cv2
import numpy as np
from detector.TensorFlowDeepAutoencoder.code.ae .utils.data import fill_feed_dict_ae, read_data_sets_pretraining
from detector.TensorFlowDeepAutoencoder.code.ae .utils.data import read_data_sets, fill_feed_dict,fill_feed_dict_1 
from detector.TensorFlowDeepAutoencoder.code.ae .utils.flags import FLAGS
from detector.TensorFlowDeepAutoencoder.code.ae .utils.eval import loss_supervised, evaluation, do_eval_summary,do_eval
from detector.TensorFlowDeepAutoencoder.code.ae .utils.utils import tile_raster_images


sess=None
y =None
input_pl=None

sess1=None
y1 =None
input_pl1=None

label =['-','+']
label.sort()

def predict_char(img):
    global sess1
    global y1
    global input_pl1

    tf.reset_default_graph()
    if(sess1==None) :

        ckpt = tf.train.get_checkpoint_state("model_sps_2017-09-27-12:55:55")
        sess1 = tf.Session()
        saver = tf.train.import_meta_graph('model_sps_2017-10-03-07:57:35/model.meta')

        saver.restore(sess1, ckpt.model_checkpoint_path)

        with sess1.as_default():


            # print(sess.run(tf.report_uninitialized_variables()))
            ae_shape = [784, 100, 75, 50, 67]
            #ae_shape = [784, 2000, 2000, 2000, 10]
            ae = AutoEncoder(ae_shape, sess,True)

            v1 = ae.get_variables_to_init(1)


            v2 = ae.get_variables_to_init(2)


            v3 = ae.get_variables_to_init(3)

            input_pl1 = tf.placeholder(tf.float32, shape=(1,
                                                 FLAGS.image_pixels),name='input_pl')
            sup_net = ae.supervised_net(input_pl1)
            y1 = tf.nn.softmax(sup_net)




    data = []

    img = cv2.resize(img,(28,28))

    data.append(img)
    data = np.array(data)
    data = data.reshape(len(data), 784)
    data = data.astype(np.float32)
    data = np.multiply(data, 1.0 / 255.0)
    
    feed = fill_feed_dict_1(data, input_pl1, noise=False)
    predictions = y1.eval(session=sess1, feed_dict=feed)
    i = np.argmax(predictions)
    score =predictions[0][i]
    #print(predictions[0][i])

    return i,score




def predict(img):
    global sess
    global y
    global input_pl
    global label
    if(sess==None) :
        print("res")
        ckpt = tf.train.get_checkpoint_state("detector/TensorFlowDeepAutoencoder/code/ae/model_sps_20170929054409")
        sess = tf.Session()
        saver = tf.train.import_meta_graph('detector/TensorFlowDeepAutoencoder/code/ae/model_sps_20170929054409/model.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)

        with sess.as_default():


            # print(sess.run(tf.report_uninitialized_variables()))
            ae_shape = [784, 100, 75, 50, 2]
            #ae_shape = [784, 2000, 2000, 2000, 10]
            ae = AutoEncoder(ae_shape, sess,True)

            v1 = ae.get_variables_to_init(1)


            v2 = ae.get_variables_to_init(2)


            v3 = ae.get_variables_to_init(3)

            input_pl = tf.placeholder(tf.float32, shape=(1,
                                                 FLAGS.image_pixels),name='input_pl')
            sup_net = ae.supervised_net(input_pl)
            y = tf.nn.softmax(sup_net)


    
    with sess.as_default():
            data = []

            img = cv2.resize(img,(28,28))

            data.append(img)
            data = np.array(data)
            data = data.reshape(len(data), 784)
            data = data.astype(np.float32)
            data = np.multiply(data, 1.0 / 255.0)
            
            feed = fill_feed_dict_1(data, input_pl, noise=False)
            predictions = y.eval(session=sess, feed_dict=feed)
            i = np.argmax(predictions)
            score =predictions[0][i]
            #print(predictions[0][i])
            #print(i)
            #print(predictions)
            return label[i],score




# class AutoEncoderTest(tf.test.TestCase):

#   def test_constructor(self):
#     ckpt = tf.train.get_checkpoint_state("model_sps_2017-08-29_11:45:25")
#     sess = tf.Session()
#     saver = tf.train.import_meta_graph('model_sps_2017-08-29_11:45:25/model.meta')
#     saver.restore(sess, ckpt.model_checkpoint_path)
#     with self.test_session() as sess:

#       ae_shape = [10, 20, 30, 2]
#       self.assertTrue(AutoEncoder(ae_shape, sess))

#   def test_get_variables(self):
#     with self.test_session() as sess:
#       ae_shape = [10, 20, 30, 2]
#       ae = AutoEncoder(ae_shape, sess)

#       with self.assertRaises(AssertionError):
#         ae.get_variables_to_init(0)
#       with self.assertRaises(AssertionError):
#         ae.get_variables_to_init(4)

#       v1 = ae.get_variables_to_init(1)
#       self.assertEqual(len(v1), 3)

#       v2 = ae.get_variables_to_init(2)
#       self.assertEqual(len(v2), 5)

#       v3 = ae.get_variables_to_init(3)
#       self.assertEqual(len(v3), 2)

#   def test_nets_1(self):

#     ckpt = tf.train.get_checkpoint_state("model_sps_2017_09_03_02_37_55")
#     sess = tf.Session()
#     saver = tf.train.import_meta_graph('model_sps_2017_09_03_02_37_55/model.meta')

#     saver.restore(sess, ckpt.model_checkpoint_path)
    
#     with sess.as_default():


#         # print(sess.run(tf.report_uninitialized_variables()))
#         ae_shape = [784, 100, 50, 50, 10]
#         #ae_shape = [784, 2000, 2000, 2000, 10]
#         ae = AutoEncoder(ae_shape, sess,True)

#         v1 = ae.get_variables_to_init(1)


#         v2 = ae.get_variables_to_init(2)


#         v3 = ae.get_variables_to_init(3)



#         input_pl = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,
#                                              FLAGS.image_pixels),
#                           name='input_pl')
#         # with self.assertRaises(AssertionError):
#         #   ae.pretrain_net(input_pl, 0)
#         # with self.assertRaises(AssertionError):
#         #   ae.pretrain_net(input_pl, 3)

#         # net1 = ae.pretrain_net(input_pl, 1)
#         # print(net1.get_shape())
#         # net2 = ae.pretrain_net(input_pl, 2)
#         # print(net2.get_shape())
#         # net3 = ae.pretrain_net(input_pl, 3)
#         # print(net3.get_shape())
#         # # # # self.assertEqual(net1.get_shape().dims[1].value, 10)
#         # # # # self.assertEqual(net2.get_shape().dims[1].value, 20)

#         # net1_target = ae.pretrain_net(input_pl, 1, is_target=True)
#         # print(net1_target.get_shape())
#         # # #self.assertEqual(net1_target.get_shape().dims[1].value, 10)
#         # net2_target = ae.pretrain_net(input_pl, 2, is_target=True)
#         # print(net2_target.get_shape())
#         # # #self.assertEqual(net2_target.get_shape().dims[1].value, 20)
#         # net3_target = ae.pretrain_net(input_pl, 3, is_target=True)
#         # print(net3_target.get_shape())
        
#         sup_net = ae.supervised_net(input_pl)
#         print(sup_net.get_shape())
#         # # self.assertEqual(sup_net.get_shape().dims[1].value, 2)



#         data = read_data_sets(FLAGS.data_dir)

#         labels_placeholder = tf.placeholder(tf.int32,
#                                           shape=FLAGS.batch_size,
#                                           name='target_pl')
#         eval_correct = evaluation(sup_net, labels_placeholder)

#         do_eval(sess,
#           eval_correct,
#           input_pl,
#           labels_placeholder,
#           data.test)

  


#   def test_nets_1(self ,ae, data_test):
#     from time import gmtime, strftime
#     import os
#     saver = tf.train.Saver()
#     folder = "model_sps_"+str(strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
#     os.mkdir(folder)
#     folder += "/model"
#     saver.save(ae.session, folder)

#     # with ae.session.as_default():
#     #     ae_shape = [10, 20, 30, 2]
#     #     ae = AutoEncoder(ae_shape, sess)

#     #     input_pl = tf.placeholder(tf.float32, shape=(100, 10))
#     #     with self.assertRaises(AssertionError):
#     #       ae.pretrain_net(input_pl, 0)
#     #     with self.assertRaises(AssertionError):
#     #       ae.pretrain_net(input_pl, 3)

#     #     net1 = ae.pretrain_net(input_pl, 1)
#     #     print(net1.get_shape())
#     #     net2 = ae.pretrain_net(input_pl, 2)
#     #     print(net2.get_shape())
#     #     self.assertEqual(net1.get_shape().dims[1].value, 10)
#     #     self.assertEqual(net2.get_shape().dims[1].value, 20)

#     #     net1_target = ae.pretrain_net(input_pl, 1, is_target=True)
#     #     print(net1_target.get_shape())
#     #     self.assertEqual(net1_target.get_shape().dims[1].value, 10)
#     #     net2_target = ae.pretrain_net(input_pl, 2, is_target=True)
#     #     print(net2_target.get_shape())
#     #     self.assertEqual(net2_target.get_shape().dims[1].value, 20)

#     #     sup_net = ae.supervised_net(input_pl)
#     #     self.assertEqual(sup_net.get_shape().dims[1].value, 2)


#     #     print(sup_net.get_shape())

#     # labels_placeholder = tf.placeholder(tf.int32,
#     #                                    shape=FLAGS.batch_size,
#     #                                    name='target_pl')
#     # eval_correct = evaluation(sup_net, labels_placeholder)
#     # do_eval(ae.session,
#     #   eval_correct,
#     #   input_pl,
#     #   labels_placeholder,
#     #   data_test)


#     input_pl = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,FLAGS.image_pixels),name='input_pl')


#     net1 = ae.pretrain_net(input_pl, 1)
#     print(net1.get_shape())
#     net2 = ae.pretrain_net(input_pl, 2)
#     print(net2.get_shape())
#     net3 = ae.pretrain_net(input_pl, 3)
#     print(net3.get_shape())

#     net1_target = ae.pretrain_net(input_pl, 1, is_target=True)
#     print(net1_target.get_shape())

#     net2_target = ae.pretrain_net(input_pl, 2, is_target=True)
#     print(net2_target.get_shape())
#     net3_target = ae.pretrain_net(input_pl, 3)
#     print(net3_target.get_shape())




#     sup_net = ae.supervised_net(input_pl)

#     print(sup_net.get_shape())

#     labels_placeholder = tf.placeholder(tf.int32,
#                                        shape=FLAGS.batch_size,
#                                        name='target_pl')
#     eval_correct = evaluation(sup_net, labels_placeholder)
#     do_eval(ae.session,
#       eval_correct,
#       input_pl,
#       labels_placeholder,
#       data_test)