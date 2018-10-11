# -*- coding: utf-8 -*-
"""
Image Catalogue

This program provides a GUI for the purpose of
assisting an editor with the categorization
of around 30TB of image files without having
to manually open, categorize, and move each
image.

Author: Ethan Dinnen
"""
import tkinter as tk
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
import re
import os
import pathlib

from label_image import *

class ImageClassifier(tk.Frame):
    """
    Classify images and sort them into folders

    This class both creates the GUI and classifies
    images with our cross-trained Tensorflow model
    """

    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the GUI and start classification

        Args:
            parent: The Tkinter window

        Returns:
            null
        """
        # Begin initializing the GUI
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.root = parent
        self.root.wm_title("Classify Image")
        src = "./TestImages/"

        self.list_images = []
        for d in os.listdir(src):
            if '.jpg' in d:
                self.list_images.append(d)

        self.frame1 = tk.Frame(self.root, width=500, height=400, bd=2)
        self.frame1.grid(row=1, column=0)
        self.frame2 = tk.Frame(self.root, width=500, height=400, bd=1)
        self.frame2.grid(row=1, column=1)

        self.canvas1 = tk.Canvas(self.frame1, height=360, width=480, background="white", bd=1, relief=tk.RAISED)
        self.canvas1.grid(row=1, column=0)
        self.canvas2 = tk.Canvas(self.frame2, height=390, width=490, bd=2, relief=tk.SUNKEN)
        self.canvas2.grid(row=1, column=0)

        claButton = tk.Button(self.root, text='Confirm', height=2, width=10, command=self.classify_obj)
        claButton.grid(row=0, column=1, padx=2, pady=2)
        nextButton = tk.Button(self.root, text='Next', height=2, width=8, command=self.next_image)
        nextButton.grid(row=0, column=0, padx=2, pady=2)
        # GUI Initialized

        # Create the folders for each category
        self.create_folders()

        # Begin classifying images
        self.counter = 0
        self.max_counter = len(self.list_images)-1
        self.next_image()

    def next_image(self):
        """
        Open the next image

        This function clears the canvases and
        opens the next image in our list_images
        array
        """
        if self.counter > self.max_counter:
            # Wow, no more images! Clear the canvases and notify editor
            self.canvas1.delete("all")
            self.canvas2.delete("all")
            self.canvas1.create_text(125, 65, fill="darkblue", font="Roboto 15", text="No more images!")
        else:
            self.im = Image.open("{}{}".format("./TestImages/", self.list_images[self.counter]))

            # Calculate how large to make the thumbnail
            if (480-self.im.size[0])<(360-self.im.size[1]):
                width = 480
                height = width*self.im.size[1]/self.im.size[0]
                self.next_step(height, width)
            else:
                height = 360
                width = height*self.im.size[0]/self.im.size[1]
                self.next_step(height, width)

    def next_step(self, height, width):
        # self.im = Image.open("{}{}".format("./TestImages/", self.list_images[self.counter]))
        # Display the image!
        self.im.thumbnail((width, height), Image.ANTIALIAS)
        self.root.photo = ImageTk.PhotoImage(self.im)
        self.photo = ImageTk.PhotoImage(self.im)

        # If this is the first image just display it
        if self.counter == 0:
            self.canvas1.create_image(0, 0, anchor = 'nw', image = self.photo)

        # Otherwise clear the canvas first and then display
        else:
            self.im.thumbnail((width, height), Image.ANTIALIAS)
            self.canvas1.delete("all")
            self.canvas1.create_image(0, 0, anchor = 'nw', image = self.photo)

        self.normalize() # Normalize the image for the classifier
        self.classify_obj() # Classify the normalized image

        self.counter += 1 # Move on to the next image

    def classify_obj(self):
        """
        Classify the image with our Tensorflow model

        This function takes in the frozen model file (.pb) and the
        label file in order to automatically classify our images
        into the categories we cross trained into our model.
        It then deletes the normalized file and creates the dropdown
        selector for the editor.
        """
        model_file = "./output_graph.pb" # The frozen model
        label_file = "./output_labels.txt" # The labels/categories
        input_height = 299
        input_width = 299
        input_mean = 0
        input_std = 255
        input_layer = "Placeholder" # The name of our input layer
        output_layer = "model" # The name of our output layer
        # Load the normalized image
        image = "{}{}".format("./TestImages/", re.sub(r'\.jpg', '', self.list_images[self.counter], flags=re.IGNORECASE) + '.norm.jpg')

        # Load the graph
        graph = load_graph(model_file)
        # Create our tensor from the image
        t = read_tensor_from_image_file(
            image,
            input_height,
            input_width,
            input_mean,
            input_std)

        # Initialize our operations
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        # Run our model on the image
        with tf.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        # Get the last five reverse-ordered sorted indices of our results
        # Results in highest to least likely 'winning' categories
        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file) # Load in our labels

        # Print our values to the canvas
        values = '';
        for i in top_k: # Loop over the winners
            values = values + str(labels[i]) + ' => ' + str(round(results[i] * 100, 2)) + '%\n'; # Display percentages for winners
        self.canvas2.delete("all")
        self.canvas2.create_text(125, 65, fill="darkblue", font="Roboto 15", text=values)

        # Remove the normalized image
        if os.path.exists(image):
            os.remove(image)

    def normalize(self):
        """
        Normalize the loaded image for the classifier to 1024 x 768
        """
        height = 1024
        width = 768
        im=cv2.imread("./TestImages/" + self.list_images[self.counter])
        im=cv2.resize(im,(height,width))
        cv2.imwrite("./TestImages/" + re.sub(r'\.jpg', '', self.list_images[self.counter], flags=re.IGNORECASE) + '.norm.jpg', im)

    def create_folders(self):
        """
        Create the category folders to deposit the images into
        """
        labels = load_labels("./output_labels.txt") # The labels/categories
        for label in labels:
            # Create the category and its parents if needed
            pathlib.Path("./categorized/" + label).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    root = tk.Tk()
    classifier = ImageClassifier(root)
    tk.mainloop()
