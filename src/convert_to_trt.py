import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

import model_config_pb2
from google.protobuf import text_format

import os
import shutil

model_store_dir = '/modelstore'
tf_model_name = 'my_tf_model'
model_name = 'my_tf_trt_model'
model_version = 0
output_name = 'fc1000/Softmax'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_graph_def = tf.GraphDef()
with tf.gfile.GFile('/modelstore/tmp_frozen_graph.pb', 'rb') as f:
  tf_graph_def.ParseFromString(f.read())
tf_graph_def = tf.graph_util.remove_training_nodes(tf_graph_def)

print('Number of nodes before conversion: {}'.format(len(tf_graph_def.node)))
trt_graph_def = trt.create_inference_graph(
  tf_graph_def,
  outputs=[output_name],
  max_batch_size=8,
  max_workspace_size_bytes=2 << 20,
  precision_mode='fp16',
  minimum_segment_size=10)

print('Number of nodes after conversion: {}'.format(len(trt_graph_def.node)))
export_model_path = '{}/{}/{}/model.graphdef'.format(model_store_dir, model_name, model_version)
if not os.path.exists(os.path.dirname(export_model_path)):
  os.makedirs(os.path.dirname(export_model_path))
with tf.gfile.GFile(export_model_path, 'wb') as f:
  f.write(trt_graph_def.SerializeToString())

model_config = model_config_pb2.ModelConfig()
with tf.gfile.GFile('{}/{}/config.pbtxt'.format(model_store_dir, tf_model_name), 'rb') as f:
  text_format.Parse(f.read(), model_config)
model_config.platform = 'tensorflow_graphdef'
model_config.name = model_name
model_config.output[0].name = output_name
with tf.gfile.GFile('{}/{}/config.pbtxt'.format(model_store_dir, model_name), 'wb') as f:
  f.write(text_format.MessageToString(model_config))

shutil.copy(
  '{}/{}/labels.txt'.format(model_store_dir, tf_model_name),
  '{}/{}/labels.txt'.format(model_store_dir, model_name)
)
