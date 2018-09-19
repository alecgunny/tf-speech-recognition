import tensorflow as tf
import multiprocessing as mp


def get_input_fn(
    dataset_path,
    labels,
    batch_size,
    num_epochs):
  def input_fn():
    dataset = tf.data.TFRecordDataset([dataset_path])
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
          default_value=""))
      }
      parsed = tf.parse_single_example(record, features)
      spec = tf.reshape(parsed['spec'], [99, 161, 1])
      label = tf.string_split(parsed['label'], delimiter="/").values[-2:-1]
      label = table.lookup(label)[0]
      return (spec, tf.one_hot(label, len(labels)))

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
