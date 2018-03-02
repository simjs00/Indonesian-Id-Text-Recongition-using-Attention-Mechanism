import tensorflow as tf

import numpy as np

import os




	 


from detector.cnn.helper_2 import img_loader as imgloader
from detector.cnn.helper_2 import model_ocr as model
import detector.cnn.config_2.default as config

#data_train_val_dir = './detector' 
#label =imgloader.get_class(data_train_val_dir)
label =['+','-']
sess = tf.InteractiveSession()
y = tf.nn.softmax(model.logits)
saver = tf.train.Saver()

if(os.path.exists("detector/cnn/model20170823194140/model.meta") ) :
		
		saver = tf.train.import_meta_graph('detector/cnn/model20170823194140/model.meta')
		saver.restore(sess,tf.train.latest_checkpoint('detector/cnn/model20170823194140/'))



def predict(img) :



	test_data = imgloader.read_data_test(img, config.image_size)

	predictions = y.eval(feed_dict={model.x:test_data})
	# prediction = tf.argmax(predictions, 1)
	i = np.argmax(predictions)

	return label[i],predictions[0][i]

import cv2
#import detector.cnn.test_model as detector_model
#import model as detector_model
import numpy as np
import detector.ocr2 as hist
#from detector.ensorFlowDeepAutoencoder.code.aeimport autoencoder_test as detector_model
import detector.TensorFlowDeepAutoencoder.code.run as detector_model

from recognizer.src import launcher as recognizer_model
import PIL.Image as PIL
import sys
cv2.setUseOptimized(True)
