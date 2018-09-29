import tensorflow as tf
import layers


def deepspeech_model(
    num_frames,
    frame_step,
    hidden_dims,
    num_classes,
    dropout):
  # input and convert from image to time series representation
  inputs = tf.keras.Input(shape=(99, 161, 1), name='spec')
  x = layers.ImageToDeepSpeech(num_frames, frame_step)(inputs)

  # transform with some dense layers
  for n, hdim in enumerate(hidden_dims[:3]):
    x = tf.keras.layers.Dropout(dropout, name='dropout_{}'.format(n))(x)
    dense = tf.keras.layers.Dense(hdim, activation='relu')
    x = tf.keras.layers.TimeDistributed(dense, name='dense_{}'.format(n))(x)

  # perform forwards and backwards recurrent layers then combine
  # hacky solution to deepspeech layer requires that forward be called first
  cell = layers.DeepSpeechCell(hidden_dims[3])
  forward = tf.keras.layers.RNN(cell, return_sequences=False, name='forward_rnn')(x)
  backward = tf.keras.layers.RNN(cell, return_sequences=False, go_backwards=True, name='backward_rnn')(x)
  x = tf.keras.layers.Add(name='rnn_combiner')([forward, backward])

  # transform with more dense layers (not time distributed this time)
  for n, hdim in enumerate(hidden_dims[4:]):
    x = tf.keras.layers.Dropout(dropout, name='dropout_{}'.format(n+3))(x)
    x = tf.keras.layers.Dense(hdim, activation='relu', name='dense_{}'.format(n+3))(x)

  # produce output
  x = tf.keras.layers.Dropout(dropout, name='dropout_labels')(x)
  x = tf.keras.layers.Dense(num_classes, activation='softmax', name='labels')(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

