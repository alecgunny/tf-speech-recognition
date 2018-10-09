FROM tensorflow/tensorflow:nightly-gpu-py3

RUN apt-get update && \
  apt-get install -y --no-install-recommends p7zip-full ffmpeg git && \
  pip install kaggle pandas RISE jupyter_contrib_nbextensions && \
  rm -rf /var/lib/apt/lists/*

# Notebook specific installations
RUN jupyter contrib nbextension install --sys-prefix && \
  jupyter nbextension install rise --py --sys-prefix && \
  jupyter nbextension enable splitcell/splitcell --sys-prefix 

RUN mkdir /work/ && cd /work/ && git clone https://github.com/alecgunny/tf-speech-recognition.git

EXPOSE 8888
EXPOSE 6006

VOLUME /data/ /work/.kaggle/

ENV DATA_DIR=/data/ KAGGLE_CONFIG_DIR=/work/.kaggle/ SHELL=/bin/bash PASSWORD=nvidia

WORKDIR /work/tf-speech-recognition/

ENTRYPOINT ./preproc/preproc.sh && jupyter-notebook --allow-root --ip=0.0.0.0
