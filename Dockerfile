ARG tag=18.12-py3
FROM nvcr.io/nvidia/tensorflow:$tag as base
ENV DATA_DIR=/data KAGGLE_CONFIG_DIR=/tmp/.kaggle MODEL_DIR=/tmp/model SHELL=/bin/bash

# need to install fixuid tool to change ownership of files inside container at runtime
# first create a docker user and chown relevant directories to it
RUN addgroup --gid 1000 docker && \
    adduser --uid 1000 --ingroup docker --home /home/docker --disabled-password --gecos "" docker && \
    mkdir /.local $MODEL_DIR && \
    chown docker:docker /.local $MODEL_DIR

RUN USER=docker && \
    GROUP=docker && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\npaths:\n  - /home/docker\n  - /.local\n  - $MODEL_DIR" > /etc/fixuid/config.yml

# needed to wait to chown $MODEL_DIR before declaring this command
# because once a volume is created by a VOLUME step, it can't be
# changed by any subsequent build steps
VOLUME $DATA_DIR $KAGGLE_CONFIG_DIR $MODEL_DIR

# going a little overboard on build targets for illustrative purposes
# auxillary build target for monitoring with tensorboard
FROM base as tensorboard
EXPOSE 6006
ENTRYPOINT /bin/bash -c "fixuid -q && tensorboard --logdir $MODEL_DIR --host=0.0.0.0"

# python and jupyter specific installs for main base
# separated because tensorboard doesn't need these
FROM base as jupyter
RUN apt-get update && \
  apt-get install -y --no-install-recommends p7zip-full ffmpeg && \
  pip install kaggle pandas RISE jupyter_contrib_nbextensions && \
  jupyter contrib nbextension install --sys-prefix && \
  jupyter nbextension install rise --py --sys-prefix && \
  jupyter nbextension enable splitcell/splitcell --sys-prefix && \
  rm -rf /var/lib/apt/lists/*
WORKDIR /home/docker
USER docker:docker

# preprocessing container
FROM jupyter as preproc
COPY --chown=docker:docker preproc /home/docker/preproc/
ENTRYPOINT /bin/bash -c "fixuid -q && ./preproc/preproc.sh"

# notebook container
FROM preproc as main
RUN rm -r /home/docker/preproc
COPY --chown=docker:docker img Slideshow.ipynb /home/docker/
EXPOSE 8888
ENTRYPOINT /bin/bash -c  "fixuid -q && jupyter notebook --allow-root --ip=0.0.0.0 --NotebookApp.token=''"
