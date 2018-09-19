import tensorflow as tf
import layers


class DropoutThenDense(tf.keras.Model):
  def __init__(self, units, dropout=0.05, tdd=False, **kwargs):
    super(DropoutThenDense, self).__init__(**kwargs)
    self.dropout = tf.keras.layers.Dropout(dropout)
    self.dense = tf.keras.layers.Dense(units, activation='relu')
    if tdd:
      self.dense = tf.keras.layers.TimeDistributed(self.dense)
    self.units = units
    self.tdd = tdd

  def call(self, inputs):
    x = self.dropout(inputs)
    return self.dense(x)

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.units
    return shape


class DeepSpeechRNN(tf.keras.Model):
  def __init__(self, units, **kwargs):
    super(DeepSpeechRNN, self).__init__(**kwargs)
    # forward_cell = layers.DeepSpeechCell(units)
    # backward_cell = layers.DeepspeechCell(units, kernel=forward_cell.kernel, bias=forward_cell.bias)

    cell = layers.DeepSpeechCell(units)
    self.forward = tf.keras.layers.RNN(cell, return_sequences=False)
    self.backward = tf.keras.layers.RNN(cell, return_sequences=False, go_backwards=True)
    self.add = tf.keras.layers.Add()
    self.units = units

  def call(self, inputs):
    xf = self.forward(inputs)
    xb = self.backward(inputs)
    return self.add([xf, xb])

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    return [shape[0], units]


def deepspeech_model(
    num_frames,
    frame_step,
    hidden_dims,
    num_classes,
    dropout):
  inputs = tf.keras.Input(shape=(99, 161, 1), name='spec')
  x = layers.ImageToDeepSpeech(num_frames, frame_step)(inputs)

  for n, hdim in enumerate(hidden_dims[:3]):
    x = tf.keras.layers.Dropout(dropout, name='dropout_{}'.format(n))(x)
    dense = tf.keras.layers.Dense(hdim, activation='relu')
    x = tf.keras.layers.TimeDistributed(dense, name='dense_{}'.format(n))(x)

  cell = layers.DeepSpeechCell(hidden_dims[3])
  forward = tf.keras.layers.RNN(cell, return_sequences=False, name='forward_rnn')(x)
  backward = tf.keras.layers.RNN(cell, return_sequences=False, go_backwards=True, name='backward_rnn')(x)
  x = tf.keras.layers.Add(name='rnn_combiner')([forward, backward])

  for n, hdim in enumerate(hidden_dims[4:]):
    x = tf.keras.layers.Dropout(dropout, name='dropout_{}'.format(n+3))(x)
    x = tf.keras.layers.Dense(hdim, activation='relu', name='dense_{}'.format(n+3))(x)
  x = tf.keras.layers.Dense(num_classes, activation='softmax', name='labels')(x)
  return tf.keras.Model(inputs=inputs, outputs=x)


class DeepSpeechModel(tf.keras.Model):
  def __init__(
      self,
      num_frames,
      frame_step,
      hidden_dims,
      num_classes,
      dropout=0.05,
      **kwargs):
    super(DeepSpeechModel, self).__init__(name='DeepSpeech', **kwargs)

    # self.roll_input = layers.ImageToDeepSpeech(num_frames, frame_step, input_shape=(99, 161, 1), name='spec')
    self.dense_1 = DropoutThenDense(hidden_dims[0], dropout, tdd=True, name='tdd_1', input_shape=(99, 161, 1))
    self.dense_2 = DropoutThenDense(hidden_dims[1], dropout, tdd=True, name='tdd_2')
    self.dense_3 = DropoutThenDense(hidden_dims[2], dropout, tdd=True, name='tdd_3')
    self.rnn = DeepSpeechRNN(hidden_dims[3], name='rnn')
    self.dense_4 = DropoutThenDense(hidden_dims[4], dropout, tdd=False, name='tdd_4')
    self.dense_5 = DropoutThenDense(hidden_dims[5], dropout, tdd=False, name='tdd_5')
    self.dense_6 = tf.keras.layers.Dense(num_classes, activation='softmax', name='labels')

  def call(self, inputs):
    # x = self.roll_input(inputs)
    x = self.dense_1(inputs)
    x = self.dense_2(x)
    x = self.dense_3(x)
    x = self.rnn(x)
    x = self.dense_4(x)
    x = self.dense_5(x)
    return self.dense_6(x)
