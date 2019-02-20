ARG tag=19.01-py3
ARG trtisclient

# need the tensorrtserver model_config protobuf python
# object. Since
FROM $trtisclient as trtis
RUN pip install /opt/tensorrtserver/pip/*.whl

FROM nvcr.io/nvidia/tensorflow:$tag

ENV MODELSTORE=/modelstore TENSORBOARD=/tensorboard KAGGLE_CONFIG_DIR=/tmp/.kaggle
VOLUME $MODELSTORE $TENSORBOARD $KAGGLE_CONFIG_DIR

RUN apt-get update && \
      apt-get install -y --no-install-recommends p7zip-full ffmpeg && \
      pip install kaggle && \
      rm -rf /var/lib/apt/lists/*

COPY --from=trtis /usr/local/lib/python3.5/dist-packages/tensorrtserver/api/model_config_pb2.py /tmp
ENV PYTHONPATH=$PYTHONPATH:/tmp
