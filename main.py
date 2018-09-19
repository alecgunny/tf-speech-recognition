import tensorflow as tf
import models
import layers
import data_utils
import utils
import argparse


def main():
  with open(FLAGS.labels, 'r') as f:
    labels = f.read().split(",")
    labels = labels[:20] + ['unknown']

  model = models.deepspeech_model(
    FLAGS.num_frames,
    FLAGS.frame_step,
    FLAGS.hidden_sizes,
    len(labels),
    0.05)
  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metric='accuracy')
  hooks = [utils.LoggerHook(100, FLAGS.batch_size*FLAGS.num_gpus)]
  #print(model.summary())

  strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=FLAGS.num_gpus)
  config = tf.estimator.RunConfig(train_distribute=strategy)
  custom_objects = {
    'DeepSpeechCell': layers.DeepSpeechCell,
    'ImageToDeepSpeech': layers.ImageToDeepSpeech
  }
  estimator = tf.keras.estimator.model_to_estimator(
    model,
    custom_objects=custom_objects,
    config=config)

  train_input_fn = data_utils.get_input_fn(
    '/data/train.tfrecords',
    labels,
    FLAGS.batch_size,
    FLAGS.num_epochs)

  eval_input_fn = data_utils.get_input_fn(
    '/data/valid.tfrecords',
    labels,
    FLAGS.batch_size*4,
    FLAGS.num_epochs)

  # train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.num_epochs)
  # eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
  # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  estimator.train(input_fn=train_input_fn, hooks=hooks)
  estimator.evaluate(input_fn=eval_input_fn)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--num_frames',
    type=int,
    default=5,
    help='number of spectrogram frames to stack next to one another in deep speech representation')

  parser.add_argument(
    '--frame_step',
    type=int,
    default=2,
    help='number of steps to jump between in deep speech representation')

  parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='batch size')

  parser.add_argument(
    '--num_epochs',
    type=int,
    default=90,
    help='number of epochs')

  parser.add_argument(
    '--num_gpus',
    type=int,
    default=1,
    help='number of gpus to distribute training over')

  parser.add_argument(
    '--labels',
    type=str,
    default='/data/labels.txt',
    help='path to comma separated labels file')

  parser.add_argument(
    '--hidden_sizes',
    type=int,
    nargs="+",
    default=[128,128,128,128,128,128],
    help='hidden sizes, must be 6')

  parser.add_argument(
    '--lr',
    type=float,
    default=0.001,
    help='learning rate')

  FLAGS = parser.parse_args()
  assert len(FLAGS.hidden_sizes) == 6

  main()
