import tensorflow as tf
#tf.app.flags.DEFINE_string('dataset_files', None, 'The directory where the dataset files are stored.')

FLAGS = tf.app.flags.FLAGS
import time

def main(_):
  #TODO:
  count = 0
  while True:
    count+=1
    time.sleep(10)
    tf.compat.v1.logging.info("test inject script! %d" % count)
      

if __name__ == "__main__":
  tf.app.run()
