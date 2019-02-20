import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import model_config_pb2
import os

model_store_dir = '/modelstore'
tf_model_name = 'my_tf_model'
model_name = 'my_tf_trt_model'
model_version = 0


tf_graph_def = tf.GraphDef()
with tf.gfile.GFile('/tmp/frozen_graph.pb', 'rb') as f:
  tf_graph_def.ParseFromString(f.read())

print('Number of nodes before conversion: {}'.format(len(tf_graph_def.node)))
trt_graph_def = trt.create_inference_graph(
  tf_graph_def,
  outputs=['fc1000/Softmax'],
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
