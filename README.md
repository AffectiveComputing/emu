Emu
======

> Emotion Mini Utility - simple API for recognizing human emotions on picture.

This project's goal is to quickly classify 7 basic human emotions (anger, contempt, fear, disgust, happiness, sadness, surprise) by picture of a human face. Our main goal is to help in research and treatment of autism, but possible applications of Emu in industry are endless. 

Setup
---

First make sure you have [Python 3.5 or newer](https://www.python.org/downloads/).

#### Packages
Install required packages.
```
pip3 install tensorflow-gpu numpy PyQt5
```
If your machine does not support CUDA.
```
pip3 install tensorflow numpy PyQt5
```
In case of trouble with installation of Tensorflow go [here](https://www.tensorflow.org/install/)

#### Application

If you want to quickly launch gui application.
```
python3 application_main.py
```
#### Training

If you want to train your own model download and unpack [Cohn-Kanade database](http://www.consortium.ri.cmu.edu/ckagree/) to /data/png.

Preprocess your database.

```
python3 preprocessing_main.py
```

Train the model.

```
python3 training_main.py
```

Status
---

Project in progress. Feedback welcome. Still improving neural net.
