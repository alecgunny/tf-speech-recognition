import tensorflow as tf
import multiprocessing as mp
import argparse


_SAMPLE_RATE = 16000


class ThroughputHook(tf.train.StepCounterHook):
  def __init__(self, batch_size, **kwargs):
    self.batch_size = batch_size
    super(ThroughputHook, self).__init__(**kwargs)

  def begin(self):
     super(ThroughputHook, self).begin()
    self._summary_tag = 'throughput'

  def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
    super(ThroughputHook, self)._log_and_record(
      elapsed_steps*self.batch_size,
      elapsed_time,
      global_step)


# TODO: add frame_length and frame_step flags
def make_spectrogram(audio):
  frame_length = FLAGS.frame_length * _SAMPLE_RATE // 1e3
  frame_step = FLAGS.frame_step * _SAMPLE_RATE // 1e3
  stfts = tf.contrib.signal.stft(
    audio,
    frame_length=tf.cast(frame_length, tf.int32),
    frame_step=tf.cast(frame_step, tf.int32),
    fft_length=tf.cast(frame_length, tf.int32))
  magnitude_spectrograms = tf.abs(stfts)
  log_offset = 1e-6
  log_magnitude_spectrograms = tf.log(magnitude_spectrograms + log_offset)
  return tf.cast(log_magnitude_spectrograms, tf.float32)


# TODO: default input and output names when using keras?
def serving_input_receiver_fn():
  audio_input_placeholder = tf.placeholder(
    dtype=tf.float32,
    shape=[-1, _SAMPLE_RATE]
    name='audio_input')
  receiver_tensors = {'audio_input': audio_input_placeholder}
  features = {'input': make_spectrogram(audio_input_placeholder)}
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def input_fn(
    dataset_path,
    labels,
    batch_size,
    num_epochs,
    input_shape,
    stats=None,
    eps=0.0001,
    buffer_size=50000):
  dataset = tf.data.TFRecordDataset([dataset_path])
  table = tf.contrib.lookup.index_table_from_tensor(
    mapping=tf.constant(labels),
    num_oov_buckets=1)

  # load in pixel wise mean and variance
  if stats is not None:
    iterator = tf.python_io.tf_record_iterator(stats)
    features = {
      'mean': tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True),
      'var': tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True)}
    parsed = tf.parse_single_example(next(iterator), features)
    mean = tf.reshape(parsed['mean'], input_shape)
    std = tf.reshape(parsed['var']**0.5, input_shape) # square rooting the variance

  def parse_spectrogram(record):
    features = {
      'spec': tf.FixedLenSequenceFeature((), tf.float32, allow_missing=True),
      'label': tf.FixedLenFeature((), tf.string, default_value="")}
    parsed = tf.parse_single_example(record, features)

    # preprocess and normalize spectrogrm
    spec = tf.reshape(parsed['spec'], input_shape) # Time steps x Frequency bins
    if stats is not None:
      spec = (spec - mean) / (std + eps)
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
  dataset.prefetch(buffer_size=None) # tf.data.experimental.AUTOTUNE)
  return dataset


def main():
  with open(FLAGS.labels, 'r') as f:
    labels = f.read().split(",")
    labels = labels[:20]

  model = tf.keras.applications.ResNet50(input_shape=FLAGS.input_shape, classes=len(labels)+1)
  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
  print('Training for {} steps'.format(max_steps))
  print(model.summary())

  config = tf.estimator.RunConfig(
    save_summary_steps=log_steps,
    save_checkpoints_secs=eval_throttle_secs,
    log_step_count_steps=log_steps,
    train_distribute=tf.contrib.distribute.MirroredStrategy(
      num_gpus=FLAGS.num_gpus,
      prefetch_on_device=True))
  estimator = tf.keras.estimator.model_to_estimator(model, config=config)

  hooks = [ThroughputHook(effective_batch_size, every_n_steps=log_steps, output_dir=estimator.model_dir)]
  train_spec = tf.estimator.TrainSpec(
    input_fn=lambda : input_fn(
      FLAGS.train_data,
      labels,
      FLAGS.batch_size,
      FLAGS.num_epochs,
      FLAGS.input_shape,
      stats=FLAGS.pixel_wise_stats),
    max_steps=max_steps,
    export_outputs={model.output.name: tf.estimator.export.ClassificationOutput},
    hooks=hooks)
  eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda : input_fn(
      FLAGS.valid_data,
      labels,
      FLAGS.batch_size*8,
      1,
      FLAGS.input_shape,
      stats=FLAGS.pixel_wise_stats),
    throttle_secs=eval_throttle_secs)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  estimator.export_saved_model(
    '{}/{}/{}'.format(FLAGS.output_dir, FLAGS.model_name, FLAGS.model_version),
    serving_input_receiver_fn)
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--num_gpus',
    type=int,
    default=1)
  parser.add_argument(
    '--train_data',
    type=str,
    default='/data/train.tfrecords')
  parser.add_argument(
    '--valid_data',
    type=str,
    default='/data/valid.tfrecords')
  parser.add_argument(
    '--pixel_wise_stats',
    type=str,
    default=None)
  parser.add_argument(
    '--labels',
    type=str,
    default='/data/labels.txt')
  parser.add_argument(
    '--input_shape',
    nargs="+",
    type=int,
    default=[99, 161])

  parser.add_argument(
    '--learning_rate',
    type=float,
    default=2e-5)
  parser.add_argument(
    '--batch_size',
    type=int,
    default=512)
  parser.add_argument(
    '--num_epochs',
    type=int,
    default=25)

  parser.add_argument(
    '--output_dir',
    type=str,
    default='/modelstore')
  parser.add_argument(
    '--model_name',
    type=str,
    default='my_tf_model')
  parser.add_argument(
    '--model_version',
    type=int,
    default=0)

  FLAGS = parser.parse_args()

  record_iterator = tf.python_io.tf_record_iterator(FLAGS.train_data)
  num_train_samples = len([record for record in record_iterator])
  effective_batch_size = FLAGS.batch_size * FLAGS.num_gpus
  max_steps = FLAGS.num_epochs*num_train_sampls // (effective_batch_size)
  eval_throttle_secs = 120
  log_steps = 10

  main()

