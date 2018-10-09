FROM tensorflow/tensorflow:nightly-gpu-py3

RUN apt-get update && \
  apt-get install -y --no-install-recommends p7zip-full ffmpeg vim && \
  pip install kaggle pandas RISE jupyter_contrib_nbextensions && \
  rm -rf /var/lib/apt/lists/*

# Notebook specific installations
RUN jupyter contrib nbextension install --sys-prefix && \
  jupyter nbextension install rise --py --sys-prefix && \
  jupyter nbextension enable splitcell/splitcell --sys-prefix 

EXPOSE 8888
EXPOSE 6006

COPY Slideshow.ipynb /workspace/
COPY img /workspace/img
COPY preproc /workspace/preproc

VOLUME /data/ /workspace/.kaggle/

ENV DATA_DIR=/data/ KAGGLE_CONFIG_DIR=/workspace/.kaggle/ SHELL=/bin/bash PASSWORD=nvidia

WORKDIR /workspace/

ENTRYPOINT ./preproc/preproc.sh && jupyter-notebook --allow-root --ip=0.0.0.0
