import tensorflow as tf
import multiprocessing as mp


def get_stats(stats_path, input_shape=None):
  if stats_path is None:
    return None, None

  dataset = tf.data.TFRecordDataset([stats])
  def parse(record):
    features = {
      'mean': tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True),
      'var': tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True)
    }
    parsed = tf.parse_single_example(record, features)
    mean = tf.reshape(parsed['mean'], input_shape)
    var = tf.reshape(parsed['var'], input_shape)
    return mean, var
  dataset = dataset.map(parse)
  iterator = dataset.make_one_shot_iterator()
  mean, var = iterator.get_next()
  with tf.Session() as sess:
    return sess.run([mean, var])
  
def get_input_fn(
        dataset_path,
        labels,
        input_shape,
        batch_size,
        num_epochs,
        mean=None,
        var=None,
        eps=0.0001,
        buffer_size=50000):
    def input_fn():
        dataset = tf.data.TFRecordDataset([dataset_path])
        table = tf.contrib.lookup.index_table_from_tensor(
            mapping=tf.constant(labels),
            num_oov_buckets=1)

        def parse_spectrogram(record):
            features = {
                'spec': tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True),
                'label': tf.FixedLenFeature((), tf.string, default_value="")
            }
            parsed = tf.parse_single_example(record, features)

            spec = tf.reshape(parsed['spec'], input_shape) # Time steps x Frequency bins
            if mean is not None:
              spec -= mean
            if var is not None:
              spec /= var**0.5 + eps
            spec = tf.expand_dims(spec, axis=2) # add channel dimension, T x F x 1

            label = tf.string_split([parsed['label']], delimiter="/").values[-2:-1]
            label = table.lookup(label)[0]
            label = tf.one_hot(label, len(labels)+1)
            return (spec, label)

        dataset = dataset.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size, num_epochs))
        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                map_func=parse_spectrogram,
                batch_size=batch_size,
                num_parallel_calls=num_cpus))
        dataset.prefetch(buffer_size=None)

        return dataset
    return input_fn
