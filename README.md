# A Trip Through the NGC TensorFlow Container
## GTC 2019
## Alec Gunny, Scott Ellis, Jeff Weiss
This demo will extend the functionality of the <a href=ngc.nvidia.com>NVIDIA GPU Cloud</a> TensorFlow container, as well as demonstrate its role in the broader NVIDIA container ecosystem. More specifically, we will
- Train a solution to the <a href="https://www.kaggle.com/c/tensorflow-speech-recognition-challenge">Kaggle TensorFlow Speech Recognition Challenge</a>
- Accelerate the trained model using NVIDIA's TensorRT library (and its integration into TensorFlow)
- Serve that accelerated model using the TensorRT Inference Server

## Building
```
$ docker build \
    -t $USER/tensorrtserver:client \
    --target trtserver_build \
    --build-arg BUILD_CLIENTS_ONLY=1 \
    --build-arg PYVER=3.5 \
    github.com/nvidia/tensorrt-inference-server#r19.01

$ docker build \
    -t $USER/tf-speech-recognition \
    --build-arg trtisclient=$USER/tensorrtserver:client \
    --build-arg tag=19.01-py3 \
    .
```
## Preprocessing
We'll leverage TensorFlow's TFRecord file format to ingest our data during training. Before we can preprocess our data and save it to a record, we need to save it locally. This is most easily done using the Kaggle command line API. Follow the instructions for getting an API key <a href="https://github.com/Kaggle/kaggle-api">here</a>. Once you have your API key json file saved in `/path/to/kaggle/json`, we can build our preprocessing container image and then run it to pull and preprocess the data and write it to a TFRecord at `/path/to/data/`.
```
$ DATA_DIR=/path/to/data
$ KAGGLE_CONFIG_DIR=/path/to/kaggle/json

$ docker run \
  --rm \
  -it \
  --name=tf-src-preproc \
  --runtime=nvidia \
  -v $DATA_DIR:/data \
  -v $KAGGLE_CONFIG_DIR:/tmp/.kaggle/ \
  -v $PWD/src:/workspace \
  -u $(id -u):$(id -g) \
  $USER/tf-speech-recognition \
  ./preproc.sh
```
## Training
Once your datasets are prepared, we can launch our training jobs.

We'll use a Docker volume called `tensorboard` to store our model checkpoints and evaluation data. This volume will be mounted to a background container running Tensorboard and exposed to port 6006 on the host so that we can monitor training progress.
```
$ docker volume create tensorboard

$ docker run \
    --rm \
    -d \
    --runtime nvidia \
    -v tensorboard:/tensorboard \
    --name tensorboard \
    -p 6006:6006 \
    $USER/tf-speech-recognition \
    tensorboard \
    --logdir=/tensorboard \
    --host=0.0.0.0
```
We'll use a different Docker volume called `modelstore` to hold our trained and exported model. We'll recycle this volume later to run the TensorRT inference server. For now, let's mount it and `tensorboard` into our training conatiner and launch a training job.
```
$ docker volume create modelstore

$ docker run \
  --rm \
  -it \
  --runtime nvidia \
  -v $PWD/src:/workspace \
  -v modelstore:/modelstore \
  -v tensorboard:/tensorboard \
  -v $DATASET_PATH:/data \
  --name tf-speech-recognition \
  $USER/tf-speech-recognition \
  python main.py \
  --train_data /data/train.tfrecords \
  --valid_data /data/valid.tfrecords \
  --pixel_wise_stats /data/stats.tfrecords \
  --labels /data/labels.txt \
  --input_shape 99 161 \
  --learning_rate 5e-6 \
  --batch_size 256 \
  --num_epochs 1 \
  --num_gpus 4 \
  --model_store_dir /modelstore \
  --model_name my_tf_model \
  --model_version 0 \
  --count 4
```

