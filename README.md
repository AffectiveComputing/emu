Emu (in progress)
======

> Emotion Mini Utility - simple API for recognizing human emotions on picture.

This project's goal is to quickly classify 7 basic human emotions (anger, contempt, fear, disgust, happiness, sadness, surprise) by picture of a human face. Our main goal is to help in research and treatment of autism, but possible applications of Emu in industry are endless.
<p align="center">
<img align="center" src="https://user-images.githubusercontent.com/23619157/28232700-7b395b88-68f2-11e7-982f-72e109b2191d.png" width="80%">
</p>

Project Structure
---

```
|-- app
|-- data
|   |-- cascades
|   |-- dataset
|   |-- logs
|-- model
|-- emu.py
|-- train.py
```

* ```app``` - modules responsible for gui.
* ```data/cascades``` - contains OpenCV face detection cascade files.
* ```data/dataset``` - this is where the dataset in .csv format should go.
* ```data/logs``` - logs generated in the training will go here.
* ```model``` - scripts related to network's model .
* ```emu.py``` - main gui script. 
* ```train.py``` - entry point for training.

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
In case of trouble with installation of Tensorflow go [here](https://www.tensorflow.org/install/).

The application also requires cv2 module. Refer to these [tutorials](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_table_of_contents_setup/py_table_of_contents_setup.html) for guidelines regarding the OpenCV installation.

#### Application

If you want to quickly launch gui application.
```
python3 emu.py
```

#### Training

If you want to train your own model.

1. Download and unpack [FER-2013 database](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) to /data/dataset.

2. Train the model.

```
python3 train.py
```

Status
---

Project in progress. Feedback welcome. Still improving neural net.
Don't mind the labels in GUI yet, they are in a wrong order.
