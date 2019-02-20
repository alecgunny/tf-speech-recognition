import tensorrtserver.api as trtis
from scipy.io.wavfile import read
import numpy as np
import os

data_dir = '/data/train/audio/'
word = 'yes'
wavfiles = os.listdir('{}/{}'.format(data_dir, word))

batch_size = 8
wavfiles = ['{}/{}/{}'.format(data_dir, word, f) for f in wavfiles[:batch_size]]
wavs = [np.array(read(f)[1]).astype('float32') for f in wavfiles][2:]
print([w.shape for w in wavs])

status_ctx = trtis.ServerStatusContext('localhost:8001', trtis.ProtocolType.from_str('grpc'), 'my_tf_model', 0)
config = status_ctx.get_server_status()
output_name = config.model_status['my_tf_model'].config.output[0].name

ctx = trtis.InferContext(
  'localhost:8001',
  trtis.ProtocolType.from_str('grpc'),
  'my_tf_model',
  0,
  0)
result = ctx.run({'audio_input': wavs[:1]}, {output_name: (trtis.InferContext.ResultFormat.CLASS, 21)}, 1)
print(result)
