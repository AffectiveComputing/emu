Emu (in progress)
======

> Emotion Mini Utility - simple API for recognizing human emotions on picture.

This project's goal is to quickly classify 7 basic human emotions (anger, contempt, fear, disgust, happiness, sadness, surprise) by picture of a human face. Our main goal is to help in research and treatment of autism, but possible applications of Emu in industry are endless.
<p align="center">
<img align="center" src="https://user-images.githubusercontent.com/23619157/30515739-b5e55c88-9b2d-11e7-81d0-836cf2013790.png" width="80%">
<img align="center" src="https://user-images.githubusercontent.com/23619157/30515740-b5e68888-9b2d-11e7-9fc9-56d379358279.png" width="80%">
<img align="center" src="https://user-images.githubusercontent.com/23619157/30515741-b5e7e2d2-9b2d-11e7-8473-d0ca0887e36d.png" width="80%">
<img align="center" src="https://user-images.githubusercontent.com/23619157/30515742-b5e973c2-9b2d-11e7-9fb6-6e9239948100.png" width="80%">
</p>

Project Structure
---

```
|-- app
|-- model
|-- data
|   |-- cascades
|   |-- dataset
|   |-- logs
|-- emu.py
|-- preprocess.py
|-- train.py
```

* ```app``` - package for GUI app
* ```model``` - package for building and training model
* ```data/cascades``` - directory for OpenCV cascade needed for face 
detection
* ```data/dataset``` - directory where the you should unpack [FER-2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) 
* ```data/logs``` - directory for all outputs of training
* ```emu.py``` - main GUI application script
* ```preprocess.py``` - main data preprocessing script
* ```train.py``` - main training script

Setup
---

First make sure you have [Python 3.5 or newer](https://www.python.org/downloads/).

#### Dependencies

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

1. Download and unpack [FER-2013 database](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) to /data/dataset directory.

2. Before you run the training for the first time, remember to preprocess the
 data.

```
python3 preprocess.py
```

3. Run the script to train the model.

```
python3 train.py
```

Status
---

Project in progress. Feedback welcome. Still improving neural net.
Don't mind the labels in GUI yet, they are in a wrong order.
