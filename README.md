Jupyter Notebook slideshow overview of solution to <a href="https://www.kaggle.com/c/tensorflow-speech-recognition-challenge">Kaggle TensorFlow Speech Recognition Challenge</a>. Leverages a custom recurrent model based off of the model outlined in Baidu's <a href="https://arxiv.org/abs/1412.5567">Deep Speech</a> paper. Meant to demonstrate the tips and tricks to most fully utilize TensorFlow on GPUs for DL research.

To build, you'll first need to get the data. This is most easily done using the Kaggle command line API. Follow the instructions for getting an API key <a href="https://github.com/Kaggle/kaggle-api">here</a>. Once you have it saved somewhere,

```
mkdir .kaggle
mv /path/to/kaggle.json .kaggle/
docker build -t $USER/tf-src docker/
./launch.sh --data_dir /path/to/save/data/to --preproc
./launch.sh --data_dir /path/to/save/data/to
```
This will build the container, build the dataset inside of it and save it on the host to `/path/to/save/data/to`, then launch the jupyter notebook server. Once all this is done, just navigate to <your machine's ip>:8888/ and enter the token printed by the notebook server. Then you should be good to launch `Slideshow.ipynb` (consider putting your browser in full screen before you do. This if F11 on Chrome).
