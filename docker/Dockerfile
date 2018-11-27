FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN apt-get update && \
  apt-get install -y --no-install-recommends p7zip-full ffmpeg git vim && \
  pip install kaggle pandas && \
  rm -rf /var/lib/apt/lists/*

#RUN mkdir /work/ && cd /work/ && git clone https://github.com/alecgunny/tf-speech-recognition.git

EXPOSE 8888
EXPOSE 6006

VOLUME /data/ /work/

ENV DATA_DIR=/data/ KAGGLE_CONFIG_DIR=/work/.kaggle/

WORKDIR /work 

ENTRYPOINT /work/src/preproc/preproc.sh &&  python /work/src/main.py
