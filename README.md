Jupyter Notebook slideshow overview of solution to <a href="https://www.kaggle.com/c/tensorflow-speech-recognition-challenge">Kaggle TensorFlow Speech Recognition Challenge</a>. Leverages a custom recurrent model based off of the model outlined in Baidu's <a href="https://arxiv.org/abs/1412.5567">Deep Speech</a> paper. Meant to demonstrate the tips and tricks to most fully utilize TensorFlow on GPUs for DL research.

To build, you'll first need to get the data. This is most easily done using the Kaggle command line API. Follow the instructions for getting an API key <a href="https://github.com/Kaggle/kaggle-api">here</a>. Once you have it saved somewhere, you can preprocess with
```
DATA_DIR=/path/to/data
KAGGLE_CONFIG_DIR=/path/to/kaggle/json

docker build -t $USER/tf-src:preproc --build-arg tag=18.12-py3 --target preprocess github.com/alecgunny/tf-speech-recognition
docker run --rm -it --name=tf-src-preproc --runtime=nvidia -v $DATA_DIR:/data -v $KAGGLE_CONFIG_DIR:/tmp/.kaggle/ -u $(id -u):$(id -g) $USER/tf-src:preproc
```
This will build the tfrecords dataset inside the container and save it on the host at `$DATA_DIR`. Once your datasets are prepared, build and launch the jupyter notebook server with
```
docker build -t $USER/tf-src --build-arg tag=18.12-py3 github.com/alecgunny/tf-speech-recognition
docker run --rm -d --name=tf-src --runtime=nvidia -v $DATA_DIR:/data -v tensorboard:/tmp/model -p 8888:8888 -u $(id -u):$(id -g) $USER/tf-src
```
You can connect to the notebook at `<your machine's ip>:8888/notebooks/Slideshow.ipynb` (consider putting your browser in full screen before you do. This if F11 on Chrome).

If we want to monitor throughput, we can build use the `tensorboard` build target in our Dockerfile and then mount the tensorboard volume we created during the previous `docker run` call (creating a whole new image is probably overkill, but is hopefully illustrative).
```
docker build -t $USER/tf-src:tensorboard --build-arg tag=18.12-py3 --target tensorboard github.com/alecgunny/tf-speech-recognition
docker run --rm -d --name=tf-src-tensorboard --runtime=nvidia -v tensorboard:/tmp/model -p 6006:6006 -u $(id -u):$(id -g) $USER/tf-src:tensorboard
```
To clean everything up
```
docker kill tf-src tf-src-tensorboard
docker volume rm tensorboard
```
Note that right now the test data is not built or preprocessed at the moment. To build it, just uncomment the appropriate line in `preproc/preproc.sh`.
