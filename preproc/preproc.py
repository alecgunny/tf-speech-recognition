import tensorflow as tf
import os
import time
import numpy as np
import argparse
import multiprocessing as mp

_SAMPLE_RATE = 16000


def read_audio(fname):
  audio_binary = tf.read_file(fname)
  waveform = tf.contrib.ffmpeg.decode_audio(
    audio_binary,
    file_format='wav',
    samples_per_second=_SAMPLE_RATE,
    channel_count=1)[:, 0]
  num_samples = tf.shape(waveform)[0]
  pad_front = (_SAMPLE_RATE - num_samples) // 2
  pad_back = (_SAMPLE_RATE - num_samples) - pad_front
  waveform = tf.pad(waveform, [[pad_front, pad_back]])
  return waveform, fname


def make_spectrogram(dataset):
  frame_length = FLAGS.frame_length * _SAMPLE_RATE // 1e3
  frame_step = FLAGS.frame_step * _SAMPLE_RATE // 1e3
  stfts = tf.contrib.signal.stft(
    dataset,
    frame_length=tf.cast(frame_length, tf.int32),
    frame_step=tf.cast(frame_step, tf.int32),
    fft_length=tf.cast(frame_length, tf.int32))
  magnitude_spectrograms = tf.abs(stfts)
  log_offset = 1e-6
  log_magnitude_spectrograms = tf.log(magnitude_spectrograms + log_offset)
  return tf.cast(log_magnitude_spectrograms, tf.float32)


def make_iterator(dataset, batch_size, table=None):
  return (dataset.apply(
    tf.contrib.data.map_and_batch(
      map_func=read_audio,
      batch_size=batch_size,
      num_parallel_calls=mp.cpu_count())
    )
    .prefetch(None)
    .make_initializable_iterator()
  )


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(array):
  if array.ndim > 1:
    array = array.ravel()
  return tf.train.Feature(float_list=tf.train.FloatList(value=array))


def build_tfrecord(
    spectrogram,
    label,
    session,
    filename,
    dataset_size=None,
    save_stats=False):
  writer = tf.python_io.TFRecordWriter(filename)
  samples_processed = 0
  batches_processed = 0
  print("Building tfrecord file {}".format(filename))
  start_time = time.time()
  while True:
    if dataset_size is not None and batches_processed % FLAGS.log_every == 0:
      time_taken = time.time() - start_time
      samples_per_second = samples_processed / time_taken
      zeroes = int(np.ceil(np.log10(dataset_size)))
      print("{}/{} samples processed; {:0.1f} samples/sec".format(
        str(samples_processed).zfill(zeroes),
        dataset_size,
        samples_per_second))

    try:
      specs, labels = session.run([spectrogram, label])
      if batches_processed == 0 and save_stats:
        mean, var = specs.sum(axis=0), (specs**2).sum(axis=0)
      elif save_stats:
        mean += specs.sum(axis=0)
        var += (specs**2).sum(axis=0)

      for spec, l in zip(specs, labels):
        feature = {
          'spec': _float_feature(spec),
          'label': _bytes_feature(tf.compat.as_bytes(l))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
      samples_processed += len(specs)
      batches_processed += 1
    except tf.errors.OutOfRangeError:
      break

  writer.close()
  if save_stats:
    mean /= samples_processed
    var /= samples_processed

    dirname = os.path.dirname(filename)
    np.save(os.path.join(dirname, 'mean.npy'), mean)
    np.save(os.path.join(dirname, 'var.npy'), std - mean**2)

def main():
  dataset_path = FLAGS.dataset_path
  test_words = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
  aux_words = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"]
  full_word_list = os.listdir(os.path.join(dataset_path, 'train', 'audio'))
  del full_word_list[full_word_list.index('_background_noise_')]

  # our order will go:
  # 1. the 10 words to classify on the test set: test_words
  # 2. words that aren't one of the test set words but have a "regular" class representation
  # 3. words that aren't in the test set and are underrepresented in the training set: aux words
  # this will let us to more easily clip labels and train on reduced sets of labels
  words = [word for word in test_words]
  words += [word for word in full_word_list if word not in (aux_words  + test_words)]
  words += aux_words

  with open(os.path.join(dataset_path, 'train', 'validation_list.txt'), 'r') as f:
    validation_files = f.read().split("\n")[:-1]
    validation_files = [os.path.join(dataset_path, 'train', 'audio', i) for i in validation_files]

  with open(os.path.join(dataset_path, 'train', 'testing_list.txt'), 'r') as f:
    pseudo_test_files = f.read().split("\n")[:-1]
    pseudo_test_files = [os.path.join(dataset_path, 'train', 'audio', i) for i in pseudo_test_files]

  train_files = []
  for word in words:
    for fname in os.listdir(os.path.join(dataset_path, 'train', 'audio', word)):
      filename = os.path.join(dataset_path, 'train', 'audio', word, fname)
      if filename not in validation_files and filename not in pseudo_test_files:
        train_files.append(filename)

  test_files = os.listdir(os.path.join(dataset_path, 'test', 'audio'))

  train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
  valid_dataset = tf.data.Dataset.from_tensor_slices(validation_files)
  ptest_dataset = tf.data.Dataset.from_tensor_slices(pseudo_test_files)
  test_dataset = tf.data.Dataset.list_files('{}/audio/*'.format(dataset_path))

  mapping_strings = tf.constant(words)
  table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings)

  train_iterator = make_iterator(train_dataset, FLAGS.batch_size, table)
  valid_iterator = make_iterator(valid_dataset, FLAGS.batch_size, table)
  ptest_iterator = make_iterator(ptest_dataset, FLAGS.batch_size, table)
  test_iterator = make_iterator(test_files, FLAGS.batch_size, None)

  train_audio, train_labels = train_iterator.get_next()
  valid_audio, valid_labels = valid_iterator.get_next()
  ptest_audio, ptest_labels = ptest_iterator.get_next()
  test_audio, test_labels = test_iterator.get_next()

  train_spectrograms = make_spectrogram(train_audio)
  valid_spectrograms = make_spectrogram(valid_audio)
  ptest_spectrograms = make_spectrogram(ptest_audio)
  test_spectrograms = make_spectrogram(test_audio)

  sess = tf.Session()
  tf.tables_initializer().run(session=sess)
  sess.run([i.initializer for i in [train_iterator, valid_iterator, ptest_iterator]])#, test_iterator]])

  build_tfrecord(
    train_spectrograms,
    train_labels,
    sess,
    os.path.join(dataset_path, 'train.tfrecords'),
    len(train_files),
    save_stats=True)

  build_tfrecord(
    valid_spectrograms,
    valid_labels,
    sess,
    os.path.join(dataset_path, 'valid.tfrecords'),
    len(validation_files))

  build_tfrecord(
    ptest_spectrograms,
    ptest_labels,
    sess,
    os.path.join(dataset_path, 'ptest.tfrecords'),
    len(pseudo_test_files))

  build_tfrecord(
    test_spectrograms,
    test_labels,
    sess,
    os.path.join(dataset_path, 'test.tfrecords'),
    len(test_files))

  with open(os.path.join(dataset_path, 'labels.txt'), 'w') as f:
    f.write(','.join(words))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--dataset_path',
    type=str,
    default='/data/',
    help='path to data')

  parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='number of samples to process at once')
  
  parser.add_argument(
    '--log_every',
    type=int,
    default=50,
    help='batches between print logging')

  parser.add_argument(
    '--frame_length',
    type=int,
    default=20,
    help="length of spectrogram FFT in ms")

  parser.add_argument(
    '--frame_step',
    type=int,
    default=10,
    help="time between FFT windows in ms")

  FLAGS = parser.parse_args()
  main()
