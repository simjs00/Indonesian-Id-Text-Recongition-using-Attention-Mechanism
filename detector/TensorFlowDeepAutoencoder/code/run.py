import os
import shutil

from detector.TensorFlowDeepAutoencoder.code.ae.utils.flags import FLAGS, home_out
from detector.TensorFlowDeepAutoencoder.code.ae import autoencoder  as autoencoder
from detector.TensorFlowDeepAutoencoder.code.ae import autoencoder_test  as autoencoder_test
#from detector.TensorFlowDeepAutoencoder.code.ae.utils.start_tensorboard import start


_data_dir = FLAGS.data_dir
_summary_dir = FLAGS.summary_dir
_chkpt_dir = FLAGS.chkpt_dir


def _check_and_clean_dir(d):
  if os.path.exists(d):
    shutil.rmtree(d)
  os.mkdir(d)

def predict(img) :
  label,score= autoencoder_test.predict(img)
  return label,score



def main():




  # home = home_out('')
  # if not os.path.exists(home):
  #   os.makedirs(home)
  # if not os.path.exists(_data_dir):
  #   os.mkdir(_data_dir)

  _check_and_clean_dir(_summary_dir)
  _check_and_clean_dir(_chkpt_dir)

  # os.mkdir(os.path.join(_chkpt_dir, '1'))
  # os.mkdir(os.path.join(_chkpt_dir, '2'))
  # os.mkdir(os.path.join(_chkpt_dir, '3'))
  # os.mkdir(os.path.join(_chkpt_dir, 'fine_tuning'))

  # test = autoencoder_test.AutoEncoderTest()
  # test.test_nets_2()
  start()

  ae = autoencoder.main_unsupervised()
  autoencoder.main_supervised(ae)
if __name__ == '__main__':
    main()

