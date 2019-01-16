Jupyter Notebook slideshow overview of solution to <a href="https://www.kaggle.com/c/tensorflow-speech-recognition-challenge">Kaggle TensorFlow Speech Recognition Challenge</a>. Leverages a custom recurrent model based off of the model outlined in Baidu's <a href="https://arxiv.org/abs/1412.5567">Deep Speech</a> paper. Meant to demonstrate the tips and tricks to most fully utilize TensorFlow on GPUs for DL research.

## Preprocessing
We'll leverage TensorFlow's TFRecord file format to ingest our data during training. Before we can preprocess our data and save it to a record, we need to save it locally. This is most easily done using the Kaggle command line API. Follow the instructions for getting an API key <a href="https://github.com/Kaggle/kaggle-api">here</a>. Once you have your API key json file saved in `/path/to/kaggle/json`, you can preprocess with
```
$ DATA_DIR=/path/to/data
$ KAGGLE_CONFIG_DIR=/path/to/kaggle/json

$ docker build \
  -t $USER/tf-src:preproc \
  --build-arg tag=18.12-py3 \
  --target preproc \
  github.com/alecgunny/tf-speech-recognition

$ docker run \
  --rm \
  -it \
  --name=tf-src-preproc \
  --runtime=nvidia \
  -v $DATA_DIR:/data \
  -v $KAGGLE_CONFIG_DIR:/tmp/.kaggle/ \
  -u $(id -u):$(id -g) \
  $USER/tf-src:preproc
```
This will pull the data from kaggle, build the tfrecords dataset, and save it on the host at `$DATA_DIR`.

## Running the slideshow
Once your datasets are prepared, build and launch the jupyter notebook server with
```
$ docker build \
  -t $USER/tf-src \
  --build-arg tag=18.12-py3 \
  github.com/alecgunny/tf-speech-recognition

$ docker run \
  --rm \
  -d \
  --name=tf-src \
  --runtime=nvidia \
  -v $DATA_DIR:/data \
  -v tensorboard:/tmp/model \
  -p 8888:8888 \
  -u $(id -u):$(id -g) \
  $USER/tf-src
```
You can connect to the notebook at `<your machine's ip>:8888/notebooks/Slideshow.ipynb` (consider putting your browser in full screen before you do. This is F11 on Chrome).

## Monitoring
Note that we created a `tensorboard` volume where we will save our model checkpoints and training statistics. (To verify, try running `docker volume ls`.) We can mount this volume to the `tensorboard` build target in our Dockerfile to observe the impact of changes in our notebook on throughput and accuracy. (A separate container and image for this use case is massive overkill, but is hopefully illustrative for the use of build targets and volumes).
```
$ docker build \
  -t $USER/tf-src:tensorboard \
  --build-arg tag=18.12-py3 \
  --target tensorboard \
  github.com/alecgunny/tf-speech-recognition

$ docker run \
  --rm \
  -d \
  --name=tf-src-tensorboard \
  --runtime=nvidia \
  -v tensorboard:/tmp/model \
  -p 6006:6006 \
  -u $(id -u):$(id -g) \
  $USER/tf-src:tensorboard
```

## Clean up
```
$ docker kill tf-src tf-src-tensorboard
$ docker volume rm tensorboard
```
Note that the test data is not built or preprocessed at the moment. To build it, just uncomment the appropriate line in `preproc/preproc.sh`.
