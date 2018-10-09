Jupyter Notebook slideshow overview of solution to <a href="https://www.kaggle.com/c/tensorflow-speech-recognition-challenge">Kaggle TensorFlow Speech Recognition Challenge</a>. Leverages a custom recurrent model based off of the model outlined in Baidu's <a href="https://arxiv.org/abs/1412.5567">Deep Speech</a> paper. Meant to demonstrate the tips and tricks to most fully utilize TensorFlow on GPUs for DL research.

To build, you'll first need to get the data. This is most easily done using the Kaggle command line API. Follow the instructions for getting an API key <a href="https://github.com/Kaggle/kaggle-api">here</a>. Once you have it saved somewhere,

```
DATA_DIR=/path/to/data/dir
KAGGLE_CONFIG_DIR=/path/to/kaggle/json/dir

docker build -t $USER/tf-src .
docker run --rm --runtime=nvidia -v $DATA_DIR:/data -v $KAGGLE_CONFIG_DIR:/workspace/.kaggle/ -p 8888:8888 -p 6006:6006 $USER/tf-src
```
This will build the container, build the dataset inside of it (assuming it needs to be built) and save it on the host to `/path/to/save/data/to`, then launch the jupyter notebook server. You can connect to the server at <your machine's ip>:8888/ and enter the password "nvidia". Then you should be good to launch `Slideshow.ipynb` (consider putting your browser in full screen before you do. This if F11 on Chrome).

Note that right now the test data is not built or preprocessed because the slideshow only needs training set data and I'm having difficulties with the size of the test set on the cluster this is being built on. To build it, just uncomment the appropriate lines in `preproc/preproc.sh`.
