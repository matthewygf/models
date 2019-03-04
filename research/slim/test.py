import tensorflow as tf
from tensorflow.python.ops import array_ops

tf.app.flags.DEFINE_string('dataset_files', None, 'The directory where the dataset files are stored.')

FLAGS = tf.app.flags.FLAGS

def parse_fn(data):
  """to parse data"""
  # from slim/datasets/cifar10
  keys_to_features = {
      "image/encoded": tf.FixedLenFeature((), tf.string, default_value=''),
      "image/format": tf.FixedLenFeature((), tf.string, default_value='png'),
      "image/class/label": tf.FixedLenFeature(
        [], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }
  parsed = tf.parse_single_example(data, keys_to_features)
  image = tf.io.decode_image(parsed["image/encoded"], channels=3)
  label = parsed["image/class/label"] 
 
  return image, label

def main(_):
  files = tf.data.Dataset.list_files(FLAGS.dataset_files)
  # cycle_length, process 10 at a time.
  train_dataset = files.apply(tf.data.experimental.parallel_interleave(
    tf.data.TFRecordDataset, cycle_length=2))
  train_dataset = train_dataset.map(map_func=parse_fn)
  train_dataset = train_dataset.batch(64)
  iterator = tf.data.Iterator.from_structure(
      train_dataset.output_types, train_dataset.output_shapes
    )
  next_element = iterator.get_next()
  train_init_op = iterator.make_initializer(train_dataset) 

  sess = tf.Session()
  sess.run(train_init_op)
  next = sess.run(next_element)
  
  for i in range(len(next)):
    print(next[i].shape)
      

if __name__ == "__main__":
  tf.app.run()
