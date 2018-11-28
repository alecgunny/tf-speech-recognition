Solution to <a href="https://www.kaggle.com/c/tensorflow-speech-recognition-challenge" >Kaggle TensorFlow Speech Recognition Challenge</a>. Leverages a custom recurrent model based off of the model outlined in Baidu's <a href="https://arxiv.org/abs/1412.5567" >Deep Speech</a> paper to demonstrate how best to utilize TensorFlow on GPUs by leveraging the Keras, Estimator, Dataset, and Distribution APIs.

To build, you'll first need to get the data. This is most easily done using the Kaggle command line API. Follow the instructions for getting an API key <a href="https://github.com/Kaggle/kaggle-api">here</a>. Once you have it saved somewhere,

```
DATA_DIR=/path/to/save/data/
KAGGLE_CONFIG_DIR=/path/to/api/key/.kaggle/
MODEL_DIR=/path/to/save/model/

docker build -t $USER/tf-src docker/
docker run --rm -it --runtime=nvidia -v $DATA_DIR:/data -v $PWD:/work -v $KAGGLE_CONFIG_DIR:/work/.kaggle/ $USER/tf-src /work/src/preproc/preproc.sh
docker run --rm -it --runtime=nvidia -v $DATA_DIR/data -v $PWD:/work $USER/tf-src python /work/src/main.py
```
To see  what command line options you can supply, run the last command with `-h` at the end. In particular, if you want to save out the model, or monitor performance with tensorboard, mount your `$MODEL_DIR` into the container and specify the mounted directory:
```
docker run --rm -d --runtime=nvidia -v $DATA_DIR/data -v $PWD:/work -v $MODEL_DIR:/tmp/model $USER/tf-src python /work/src/main.py --model_dir /tmp/model/
docker run --rm -v $MODEL_DIR:/tmp/model -p 6006:6006 $USER/tf-src tensorboard --logdir=/tmp/model --host=0.0.0.0
```
Note that right now the test data is not built or preprocessed because I'm having difficulties with the size of the test set on the cluster this is being built on. To build it, just uncomment the appropriate lines in `preproc/preproc.sh`.
