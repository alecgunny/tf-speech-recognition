import argparse


def parse_args():
  parser = argparse.ArgumentParser()

  # data args
  parser.add_argument(
    "--train_data",
    type=str,
    default="/data/train.tfrecords",
    help="Path to training tfrecords dataset")
  parser.add_argument(
    "--valid_data",
    type=str,
    default="/data/valid.tfrecords",
    help="Path to validation tfrecords dataset")
  parser.add_argument(
    "--labels",
    type=str,
    default="/data/labels.txt",
    help="Path to text file containing comma separated labels")
  parser.add_argument(
    "--input_shape",
    type=int,
    nargs="+",
    default=[99, 161],
    help="Space separated list of spectrogram dimensions in [T, F] order")
  parser.add_argument(
    "--stats",
    type=str,
    default=None,
    help="Path to tfrecords file containing pixel-wise mean and variance")
  parser.add_argument(
    "--num_train_examples",
    type=int,
    default=51088,
    help="Number of examples in training data set. 
      Helps to make sure the proper number of epochs are iterated through")

  # model args
  parser.add_argument(
    "--num_frames",
    type=int,
    default=7,
    help="Number of contiguous spectrogram timesteps to concatenate")
  parser.add_argument(
    "--frame_step",
    type=int,
    default=2,
    help="Number of timesteps between spectrogram windowing")
  parser.add_argument(
    "--hidden_sizes",
    type=int,
    help="Number of contiguous spectrogram timesteps to concatenate")
  parser.add_argument(
    "--frame_step",
    type=int,
    default=2,
    help="Number of timesteps between spectrogram windowing")
  parser.add_argument(
    "--hidden_sizes",
    type=int,
    default=[1024, 2048, 2048, 1024, 2048],
    nargs="+",
    help="hidden sizes of the model separated by spaces. Fourth will be the recurrent dimension")

  # training args
  parser.add_argument(
    "--batch_size",
    type=int,
    default=512,
    help="Batch size per gpu")
  parser.add_argument(
    "--num_gpus",
    type=int,
    default=1,
    help="number of gpus to distribute over")
  parser.add_argument(
    "--num_epochs",
    type=int,
    default=50,
    help="number of training epochs")
  parser.add_argument(
    "--lr",
    type=float,
    default=2e-5,
    help="learning rate")
  parser.add_argument(
    "--dropout",
    type=float,
    default=0.05,
    help="dropout drop probability")

  # scaffolding
  parser.add_argument(
    "--log_steps",
    type=int,
    default=10,
    help="how frequently to record  logging stats")
  parser.add_argument(
    "--eval_throttle_secs",
    type=int,
    default=120,
    help="time between evaluations on validation data")
  parser.add_argument(
    "--model_dir",
    type=str,
    default=None,
    help="Where to save checkpoints, model, etc.")

  FLAGS = parser.parse_args()
  assert len(FLAGS.input_shape) == 2
  FLAGS.input_shape = tuple(FLAGS.input_shape)
  return FLAGS

