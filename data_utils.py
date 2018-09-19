import tensorflow as tf
import multiprocessing as mp


def get_input_fn(
    dataset_path,
    num_classes,
    batch_size,
    num_epochs):
  def input_fn():
    dataset = tf.data.TFRecordDataset([dataset_path])
    def parse_spectrogram(record):
      features = {
        'train/spec': tf.FixedLenSequenceFeature((),
          tf.float32,
          allow_missing=True,
          default_value=tf.zeros([], dtype=tf.float32)),
        'train/label': tf.FixedLenFeature((),
          tf.int64,
          default_value=tf.zeros([], dtype=tf.int64))
      }
      parsed = tf.parse_single_example(record, features)
      spec = tf.reshape(parsed['train/spec'], [99, 161, 1])
      label = tf.clip_by_value(parsed['train/label'], 0, num_classes)
      return (spec, tf.one_hot(label, num_classes))

    dataset = dataset.apply(
      tf.contrib.data.shuffle_and_repeat(10000, num_epochs))
    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
        map_func=parse_spectrogram,
        batch_size=batch_size,
        num_parallel_calls=mp.cpu_count()
    ))
    dataset.prefetch(buffer_size=None)
    return dataset
  return input_fn
