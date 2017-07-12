"""
This module contains class responsible for graphics user interface.
"""


import tkinter as tk
from os import path
from tkinter.filedialog import askopenfilename

import numpy as np
from PIL import Image, ImageTk

from model.const import CLASSES_COUNT
from model.model import Model
from model.predictor import Predictor
from preprocessing import image_utilities
from preprocessing import data_set_preparing
from model import const

__author__ = ["Paweł Kopeć", "Michał Górecki"]


class Window(tk.Frame):
    """
    Class, which represents modified tkinter gui frame.
    """

    # Window dimension constants.
    __WINDOW_WIDTH = 800
    __WINDOW_HEIGHT = 600
    # Image label constant size.
    __PREVIEW_WIDTH = 500
    __PREVIEW_HEIGHT = 500

    def __init__(self, master=None):
        if master is None:
            master = tk.Tk()

        super().__init__(master)

        self.master.wm_title("Emotion recognition")
        self.master.geometry(
            "{}x{}".format(self.__WINDOW_WIDTH, self.__WINDOW_HEIGHT)
        )
        self.__setup_layout()
        self.__create_widgets()
        self.__set_results([0.0 for i in range(CLASSES_COUNT)])

        self.__predictor = Predictor(const.MODEL_FILE, const.META_FILE)
        self.__image_path = None

    def __setup_layout(self):
        """
        Setups frames necessary to create layout and configures their sizes.
        :return: -
        """
        # Configure two columns of main frame.
        self.grid(row=0, column=0)
        self.grid_columnconfigure(0, minsize=0.75 * self.__WINDOW_WIDTH)
        self.grid_columnconfigure(1, minsize=0.25 * self.__WINDOW_WIDTH)
        self.grid_rowconfigure(0, minsize=self.__WINDOW_HEIGHT)
        # Create and configure side toolbox.
        self.__toolbox = tk.Frame(self)
        self.__toolbox.grid(column=1, sticky="news")
        self.__toolbox.grid_rowconfigure(
            0, minsize=0.25 * self.__WINDOW_HEIGHT
        )
        self.__toolbox.grid_rowconfigure(
            1, minsize=0.75 * self.__WINDOW_HEIGHT
        )
        self.__toolbox.grid_columnconfigure(
            0, minsize=0.25 * self.__WINDOW_WIDTH
        )
        # Create and configure buttons frame.
        self.__buttons = tk.Frame(self.__toolbox)
        self.__buttons.grid(row=0, sticky="news")
        for i in range(2):
            self.__buttons.grid_rowconfigure(
                i, minsize=0.5 * 0.25 * self.__WINDOW_HEIGHT
            )
        self.__buttons.grid_columnconfigure(
            0, minsize=0.25 * self.__WINDOW_WIDTH
        )
        # Create and configure results frame.
        self.__results = tk.Frame(self.__toolbox)
        self.__results.grid(row=1, sticky="news")
        for i in range(2):
            self.__results.grid_columnconfigure(
                i, minsize=0.5 * 0.25 * self.__WINDOW_WIDTH
            )
        for i in range(CLASSES_COUNT):
            self.__results.grid_rowconfigure(
                i,
                minsize=1 / CLASSES_COUNT * 0.75
                        * 0.5 * self.__WINDOW_HEIGHT
            )

    def __create_widgets(self):
        """
        Create necessary gui widgets.
        :return: -
        """
        # Create image label and fill it with empty image.
        self.__image = tk.Label(self, borderwidth=2, relief="solid")
        empty_image = ImageTk.PhotoImage(
            Image.new(
                "RGB", (self.__PREVIEW_WIDTH, self.__PREVIEW_HEIGHT),
                (255, 255, 255)
            )
        )
        self.__image.config(image=empty_image)
        self.__image.image = empty_image
        self.__image.grid(row=0, column=0)
        # Setup control buttons.
        self.__open_image_button = tk.Button(
            self.__buttons, text="Open Image", command=self.__open_image
        )
        self.__open_image_button.grid(row=0)
        self.__analyze_button = tk.Button(
            self.__buttons, text="Analyze", command=self.__analyze_image
        )
        self.__analyze_button.grid(row=1)
        # Setup results text.
        self.__classes_headers = []
        self.__classes_results = []
        # Constant fill-in of result labels.
        HEADER_TEXT = "Class {}"
        for i in range(CLASSES_COUNT):
            header_label = tk.Label(self.__results, text=HEADER_TEXT.format(i))
            header_label.grid(row=i, column=0)
            result_label = tk.Label(self.__results)
            result_label.grid(row=i, column=1)
            self.__classes_headers.append(header_label)
            self.__classes_results.append(result_label)

    def __open_image(self):
        """
        Handle open image button press.
        :return: -
        """
        self.__image_path = askopenfilename()
        if self.__image_path and path.isfile(self.__image_path ):
            image = self.__load_image(self.__image_path )
            self.__image.config(image=image)
            self.__image.image = image

    def __load_image(self, path):
        """
        Load image from file with given path.
        :param path: path of source image file
        :return: loaded tk image
        """
        image = Image.open(path).resize(
            (self.__PREVIEW_WIDTH, self.__PREVIEW_HEIGHT), Image.ANTIALIAS
        )
        return ImageTk.PhotoImage(image)

    def __analyze_image(self):
        """
        Analyze loaded image with conv-net and display results.
        :return: -
        """
        CASCADE_PATH = "data/cascades/haarcascade_frontalface_default.xml"
        if self.__image_path:
             img = image_utilities.load_image(self.__image_path)
             cascade = image_utilities.load_cascade(CASCADE_PATH)
             input, _ = data_set_preparing.process_image(
                 cascade, img, 1.05, 5, False, False, 0.1, "grayscale", (64, 64)
             )
             input[0] = np.expand_dims(input[0], -1)
             scores = self.__predictor.infer(input)[0]
             self.__set_results(scores)


    def __set_results(self, results):
        """
        Set results labels to given values.
        :return: -
        """
        for i, j in enumerate(results):
            self.__classes_results[i]["text"] = "{:.3f}".format(j)
