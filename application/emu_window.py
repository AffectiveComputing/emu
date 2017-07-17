import os

import numpy as np
from PyQt5.QtCore import QMimeData
from PyQt5.QtGui import QPixmap, QPainter, QColor, QImage
from PyQt5.QtWidgets import QMainWindow, QPushButton, \
    QLabel, QFileDialog

from model.classifier import Classifier
from model.const import MODEL_FILE, META_FILE
from preprocessing.image_utilities import load_image, load_cascade, \
    extract_faces, resize, convert_to_colorspace

__author__ = ["Paweł Kopeć"]


class EmuWindow(QMainWindow):

    __EMOTIONS = ["Contempt", "Fear", "Happiness", "Disgust", "Sadness",
                  "Anger", "Surprise"]

    __CASCADE_PATH = "data/cascades/haarcascade_frontalface_default.xml"

    def __init__(self):
        super().__init__()

        self.__init_model()
        self.__init_window()
        self.__init_image_frame()
        self.__init_infos()
        self.__init_buttons()

    def __init_model(self):
        self.__classifier = Classifier(MODEL_FILE, META_FILE)
        self.__cascade = load_cascade(self.__CASCADE_PATH)
        self.__input = None
        #TODO error message

    def __init_window(self):
        self.setGeometry(300, 300, 900, 600)
        self.setWindowTitle('Emu')

    def __init_image_frame(self):
        self.__image = QLabel(self)
        self.__image_url = None
        self.__image_frame = QMimeData()

    def __init_infos(self):
        self.__emotions_title = QLabel(self)
        self.__emotions_title.setText('<h2>Emotions</h2>')
        self.__emotions_title.move(650, 15)

        self.__emotions_labels = [None] * len(self.__EMOTIONS)
        self.__emotions_scores = [None] * len(self.__EMOTIONS)

        for i, emotion in enumerate(self.__EMOTIONS):
            self.__emotions_labels[i] = QLabel(self)
            self.__emotions_scores[i] = QLabel(self)
            self.__emotions_labels[i].setText('<h3>' + emotion +'</h3>')
            self.__emotions_labels[i].move(655, 45 + 20 * i)
            self.__emotions_scores[i].move(785, 45 + 20 * i)

        self.__print_scores([0.0] * len(self.__EMOTIONS))

    def __init_buttons(self):
        self.__choose_button = QPushButton('Choose image', self)
        self.__choose_button.resize(self.__choose_button.sizeHint())
        self.__choose_button.clicked.connect(self.__load_image)
        self.__choose_button.move(240, 560)

        self.__check_button = QPushButton('Check emotion', self)
        self.__check_button.resize(self.__check_button.sizeHint())
        self.__check_button.clicked.connect(self.__check_image)
        self.__check_button.move(360, 560)

    def __load_image(self):
        image_path = QFileDialog.getOpenFileName()[0]
        if image_path:
            try:
                if not image_path or not os.path.exists(image_path):
                    raise FileNotFoundError("No such file {}".format(image_path))

                try:
                    image = load_image(image_path)
                except:
                    raise ValueError("Couldn't load image")

                if image is False:
                    raise ValueError("Couldn't load image")

                self.__image_path = image_path
                faces = extract_faces(image)
                if len(faces) == 0:
                    raise ValueError("Couldn't find face on an image")

                # Set face as featured image in gui.
                featured_face = resize(faces[0], (512, 512))
                self.__image.setGeometry(94, 20, 512, 512)
                q_image = self.__image_to__q_image(featured_face)
                self.__image.setPixmap(QPixmap.fromImage(q_image))
                self.__print_scores([0] * len(self.__EMOTIONS))

                # Set input for the neural net.
                self.__input = convert_to_colorspace([faces[0]], "grayscale")[0]
                self.__input = resize(self.__input, Classifier.INPUT_SIZE)
                self.__input = np.expand_dims(self.__input, -1)
            except Exception as e:
                #TODO label with error messages
                print(e)

    def __image_to__q_image(self, image):
        height, width, _ = image.shape
        bytes_per_line = 3 * width
        return QImage(image.data, width, height, bytes_per_line,
                      QImage.Format_RGB888)

    def __check_image(self):
        if self.__input is not None:
            scores = self.__classifier.infer([self.__input])[0]
            self.__print_scores(scores)
        else:
            #TODO error message
            pass

    def __print_scores(self, scores):
        for label, score in zip(self.__emotions_scores, scores):
            label.setText('<h3>' + str('%.2f'%(score * 100) + ' %</h3>'))

    def paintEvent(self, e):
        painter = QPainter()
        painter.begin(self)
        self.__draw_frame(painter)
        painter.end()

    def __draw_frame(self, painter):
        col = QColor(0, 0, 0)
        painter.setPen(col)
        painter.setBrush(QColor(0, 0, 0))
        painter.drawRect(89, 15, 522, 522)
        painter.drawRect(94, 20, 512, 512)
