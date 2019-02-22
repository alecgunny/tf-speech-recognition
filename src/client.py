import tensorrtserver.api as trtis
from scipy.io.wavfile import read
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from random import shuffle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='my_tf_model')
FLAGS = parser.parse_args()

status_ctx = trtis.ServerStatusContext(
  'localhost:8001',
  trtis.ProtocolType.from_str('grpc'),
  FLAGS.model_name,
  0)
config = status_ctx.get_server_status()
output_name = config.model_status[FLAGS.model_name].config.output[0].name
ctx = trtis.InferContext(
 'localhost:8001',
  trtis.ProtocolType.from_str('grpc'),
  FLAGS.model_name,
  0,
  0)

with open('/modelstore/{}/labels.txt'.format(FLAGS.model_name), 'r') as f:
  words = f.read().split("\n")[:-1]
data_dir = '/data/train/audio/'
batch_size = 8
max_samples = 200
for n, word in enumerate(words):
  wavfiles = os.listdir('{}/{}'.format(data_dir, word))
  shuffle(wavfiles)
  wavfiles = ['{}/{}/{}'.format(data_dir, word, f) for f in wavfiles[:max_samples]]
  wavs = [np.array(read(f)[1]).astype('float32') for f in wavfiles]
  wavs = [w for w in wavs if w.shape[0] ==  16000]
  num_batches = (len(wavs) - 1) // batch_size + 1
  y_preds = []
  for i in range(num_batches):
    inputs = wavs[i*batch_size:(i+1)*batch_size]
    result = ctx.run({'audio_input': inputs}, {output_name: (trtis.InferContext.ResultFormat.CLASS, 21)}, len(inputs))
    y_preds.extend([r[0][0] for r in result[output_name]])
  print(word)
  print(confusion_matrix([n]*len(y_preds), y_preds))
  print((np.array(y_preds) == np.array([n]*len(y_preds))).mean())

