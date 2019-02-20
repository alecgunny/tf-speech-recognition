#!/bin/bash 
python /usr/local/lib/python3.5/dist-packages/tensorflow/python/tools/freeze_graph.py \
  --input_saved_model_dir /modelstore/my_tf_model/0/model.savedmodel \
  --output_graph /tmp/frozen_graph.pb \
  --output_node_names fc1000/Softmax \
  --saved_model_tags serve \
  --input_binary

python convert_to_trt.py
