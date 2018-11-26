import tensorflow as tf


class ImageToDeepSpeech(tf.keras.layers.Layer):
  def __init__(self, num_frames, frame_step, **kwargs):
    self.num_frames = num_frames
    self.frame_step = frame_step
    super(ImageToDeepSpeech, self).__init__(**kwargs)

  def call(self, inputs):
    inputs = tf.squeeze(inputs, axis=3)
    time_slice = lambda x, i: x[:, i:(-(self.num_frames-1)+i) or None:self.frame_step]
    time_shifted_inputs = [time_slice(inputs, i) for i in range(self.num_frames)]
    return tf.concat(time_shifted_inputs, axis=2)

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    time_dim = tf.ceil((shape[1] - self.num_frames + 2) / self.frame_step)
    feature_dim = self.num_frames*shape[2]
    if self.frame_step == 1:
      time_dim += 1
    return tf.TensorShape([shape[0], time_dim, feature_dim])

  def get_config(self):
    base_config = super(ImageToDeepSpeech, self).get_config()
    base_config['num_frames'] = self.num_frames
    base_config['frame_step'] = self.frame_step
    return base_config


class DeepSpeechCell(tf.keras.layers.Layer):
  def __init__(self, state_size, **kwargs):
    self.state_size = state_size
    super(DeepSpeechCell, self).__init__(**kwargs)

  def call(self, inputs, states):
    prev_output = states[0]
    h = tf.matmul(inputs, self.kernel)
    u = tf.matmul(prev_output, self.recurrent_kernel)
    output = tf.nn.relu(h + u + self.bias)
    return output, [output]

  def build(self, input_shape):
    self.input_dim = input_shape[1]
    if self.built:
      # normally you just return. This is a hack to allow the
      # second calling of this cell to use a different recurrent
      # kernel. Not elegant but that's showbiz baby
      self.recurrent_kernel = self.backward_recurrent_kernel
      return

    self.kernel = self.add_weight(
      shape=(self.input_dim, self.state_size),
      name='kernel',
      initializer='glorot_normal')
    self.bias = self.add_weight(
      shape=(self.state_size,),
      name='bias',
      initializer='zeros')

    self.forward_recurrent_kernel = self.add_weight(
      shape=(self.state_size, self.state_size),
      name='forward_recurrent_kernel',
      initializer='glorot_normal')
    self.backward_recurrent_kernel = self.add_weight(
      shape=(self.state_size, self.state_size),
      name='backward_recurrent_kernel',
      initializer='glorot_normal')

    self.recurrent_kernel = self.forward_recurrent_kernel
    super(DeepSpeechCell, self).build(input_shape)

  def get_config(self):
    base_config = super(DeepSpeechCell, self).get_config()
    base_config['state_size'] = self.state_size
    return base_config


def deepspeech_model(
    input_shape,
    num_frames,
    frame_step,
    hidden_dims,
    num_classes,
    dropout=0.05):
  # input and convert from image to time series representation
  input = tf.keras.Input(shape=input_shape, name='spec')
  x = ImageToDeepSpeech(num_frames, frame_step)(input)

  # transform with 3 time distributed dense layers
  for n, hdim in enumerate(hidden_dims[:3]):
    x = tf.keras.layers.Dropout(dropout, name='dropout_{}'.format(n))(x)
    dense = tf.keras.layers.Dense(hdim, activation='relu')
    x = tf.keras.layers.TimeDistributed(dense, name='dense_{}'.format(n))(x)

  # perform forwards and backwards recurrent layers then combine
  # note that we're not returning sequences, so we're going from shape
  # B x T x F --> B x F
  cell = DeepSpeechCell(hidden_dims[3])
  forward = tf.keras.layers.RNN(cell, return_sequences=False, name='forward_rnn')(x)
  backward = tf.keras.layers.RNN(cell, return_sequences=False, go_backwards=True, name='backward_rnn')(x)
  x = tf.keras.layers.Add(name='rnn_combiner')([forward, backward])

  # transform with more dense layers (now not time distributed)
  for n, hdim in enumerate(hidden_dims[4:]):
    x = tf.keras.layers.Dropout(dropout, name='dropout_{}'.format(n+3))(x)
    x = tf.keras.layers.Dense(hdim, activation='relu', name='dense_{}'.format(n+3))(x)

  # produce output
  x = tf.keras.layers.Dropout(dropout, name='dropout_labels')(x)
  x = tf.keras.layers.Dense(num_classes, activation='softmax', name='labels')(x)
  return tf.keras.Model(inputs=input, outputs=x)

