import tensorflow as tf
import multiprocessing as mp
import numpy as np


def get_input_fn(
    dataset_path,
    labels,
    batch_size,
    num_epochs,
    mean='/data/mean.npy',
    std='/data/std.npy',
    eps=0.0001):
  def input_fn():
    dataset = tf.data.TFRecordDataset([dataset_path])
    mean_spec = np.load(mean)
    std_spec = np.load(std)**0.5
    table = tf.contrib.lookup.index_table_from_tensor(
      mapping=tf.constant(labels),
      num_oov_buckets=1)

    def parse_spectrogram(record):
      features = {
        'spec': tf.FixedLenSequenceFeature((),
          tf.float32,
          allow_missing=True,
          default_value=tf.zeros([], dtype=tf.float32)),
        'label': tf.FixedLenFeature((),
          tf.string,
          default_value="")
      }
      parsed = tf.parse_single_example(record, features)

      spec = tf.reshape(parsed['spec'], [99, 161])
      spec = (spec - mean_spec) / (std_spec + eps)
      spec = tf.expand_dims(spec, axis=2)

      label = tf.string_split([parsed['label']], delimiter="/").values[-2:-1]
      label = table.lookup(label)[0]
      label = tf.one_hot(label, len(labels))
      return (spec, label)

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
