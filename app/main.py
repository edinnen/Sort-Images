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
import unidecode
import json
from shutil import copyfile

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
        # Initialize some variables we will need
        self.category = '' # The selected category for the image
        self.shoot_name = '' # The name of the current image's parent directory
        self.labels = load_labels("./output_labels.txt")

        # Load categories
        if not pathlib.Path("./categories.txt").exists(): # Create our category file if it does not exist
            if pathlib.Path("./categorized").exists(): # If the editor has already categorized some images, they could have added new categories. So we want to load them
                with open("./categories.txt", 'w') as categories:
                    for category in next(os.walk('./categorized'))[1]:
                        categories.write("{}\n".format(category))
            else:
                copyfile("./output_labels.txt", "./categories.txt") # Otherwise just copy the output_labels file
        self.categories = load_labels("./categories.txt")

        # Create the folders for each category
        self.create_folders()

        # Begin initializing the GUI
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.root = parent
        self.root.wm_title("Classify Image")

        # Create the list of images to process
        src = "./TestImages/"
        path = pathlib.Path(src)
        completed = "./categorized"
        completedPath = pathlib.Path(completed)
        self.list_completed = [re.match(r'.*/(.*)/(.*)$', str(f))[2] for f in completedPath.glob('**/*.jpg') if f.is_file()] # Find the images we have already processed
        self.list_images = [f for f in path.glob('**/*.jpg') if f.is_file() and re.match(r'.*/(.*)/(.*)$', str(f))[2] not in self.list_completed] # Find all the unprocessed images

        self.frame1 = tk.Frame(self.root, width=500, height=400, bd=2)
        self.frame1.grid(row=1, column=0)

        self.canvas1 = tk.Canvas(self.frame1, height=360, width=480, background="white", bd=1)
        self.canvas1.grid(row=1, column=0)
        self.canvas2 = tk.Canvas(self.root, width=500, height=400, bd=2)
        self.canvas2.grid(row=1, column=1)
        self.frame2 = tk.Frame(self.canvas2, width=500, height=400, bd=2)
        self.canvas2.create_window(0,0,window=self.frame2,anchor='nw')

        claButton = tk.Button(self.root, text='Confirm', height=2, width=10, command=self.copy_to_category)
        claButton.grid(row=0, column=1, padx=2, pady=2)
        nextButton = tk.Button(self.root, text='Skip', height=2, width=8, command=self.next_image)
        nextButton.grid(row=0, column=0, padx=2, pady=2)
        # GUI Initialized

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
        # Clear the frame
        for widget in self.frame2.winfo_children():
            widget.destroy()
        if self.counter > self.max_counter:
            # Wow, no more images! Clear the canvases and notify editor
            self.canvas1.delete("all")
            self.canvas2.delete("all")
            self.canvas1.create_text(125, 65, fill="darkblue", font="Roboto 15", text="No more images!")
        else:
            self.im = Image.open(str(self.list_images[self.counter]))
            self.shoot_name = re.match(r'.*/(.*)/(.*)$', str(self.list_images[self.counter]), re.IGNORECASE)[1]
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
        self.create_fields() # Create the metadata editing fields for the editor

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
        input_height = 299
        input_width = 299
        input_mean = 0
        input_std = 255
        input_layer = "Placeholder" # The name of our input layer
        output_layer = "model" # The name of our output layer
        # Load the normalized image
        image = str(re.sub(r'\.jpg', '', str(self.list_images[self.counter]), flags=re.IGNORECASE) + '.norm.jpg')

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
        labels = self.labels # Load in our labels

        # Set the category as the top result by default.
        self.category = labels[top_k[0]]

        # Remove the normalized image
        if os.path.exists(image):
            os.remove(image)

    def normalize(self):
        """
        Normalize the loaded image for the classifier to 1024 x 768
        """
        height = 1024
        width = 768
        im=cv2.imread(str(self.list_images[self.counter]))
        im=cv2.resize(im,(height,width))
        cv2.imwrite(re.sub(r'\.jpg', '', str(self.list_images[self.counter]), flags=re.IGNORECASE) + '.norm.jpg', im)

    def create_fields(self):
        """
        Create the fields required for metadata entry
        """
        # Category selector
        self.selectedCategory = tk.StringVar() # String variable to hold our choice
        labels = {l for l in self.categories} # Dictionary with our category options
        self.selectedCategory.set(self.category)
        self.selectMenu = tk.OptionMenu(self.frame2, self.selectedCategory, *labels, command=self.change_category)
        self.selectMenu.config(width=35)
        self.selectMenu.grid(column=1, row=0)

        # New category creator
        self.newCategoryVar = tk.StringVar()
        self.newCategoryVar.set("None of the above. Enter new category name:")
        self.newCategory = tk.Entry(self.frame2, textvariable=self.newCategoryVar)
        self.newCategory.config(width=35)
        self.newCategory.bind("<FocusIn>", lambda args: self.newCategory.delete('0', 'end'))
        self.newCategory.grid(column=1, row=3)

        # License type selector
        self.selectedLicense = tk.StringVar()
        licenses = {'Apache 2.0', 'MIT'}
        self.selectedLicense.set('MIT') # Set the default license
        self.licenseMenu = tk.OptionMenu(self.frame2, self.selectedLicense, *licenses, command=self.change_license)
        self.licenseMenu.config(width=35)
        self.licenseMenu.grid(column=1, row=6)

        # Creative name field
        self.creativeNameVar = tk.StringVar()
        self.creativeNameVar.set("Creative Name:") # Set our placeholder
        self.creativeName = tk.Entry(self.frame2, textvariable=self.creativeNameVar) # Create the text field
        self.creativeName.config(width=35)
        self.creativeName.bind("<FocusIn>", lambda args: self.creativeName.delete('0', 'end')) # Delete placeholder text upon focus
        self.creativeName.grid(column=1, row=9)

        # Photo credit field
        self.creditVar = tk.StringVar()
        self.creditVar.set("Photo Credit:") # Set our placeholder
        self.credit = tk.Entry(self.frame2, textvariable=self.creditVar) # Create the text field
        self.credit.config(width=35)
        self.credit.bind("<FocusIn>", lambda args: self.credit.delete('0', 'end')) # Delete placeholder text upon focus
        self.credit.grid(column=1, row=12)

        # Collection field
        self.collectionVar = tk.StringVar()
        self.collectionVar.set("Collection:") # Set our placeholder
        self.collection = tk.Entry(self.frame2, textvariable=self.collectionVar) # Create the text field
        self.collection.config(width=35)
        self.collection.bind("<FocusIn>", lambda args: self.collection.delete('0', 'end')) # Delete placeholder text upon focus
        self.collection.grid(column=1, row=15)

        # Tags field
        self.tagsVar = tk.StringVar()
        self.tagsVar.set("Tags. Comma separated. E.g., tag1, tag2, tag3")
        self.tags = tk.Entry(self.frame2, textvariable=self.tagsVar)
        self.tags.config(width=35)
        self.tags.bind("<FocusIn>", lambda args: self.tags.delete('0', 'end'))
        self.tags.grid(column=1, row=18)

        # Date Collected field
        self.dateVar = tk.StringVar()
        self.dateVar.set("Date Collected: (Format like yyyy-mm-dd)")
        self.date = tk.Entry(self.frame2, textvariable=self.dateVar) # Create the text field
        self.date.config(width=35)
        self.date.bind("<FocusIn>", lambda args: self.date.delete('0', 'end')) # Delete placeholder text upon focus
        self.date.grid(column=1, row=21)

        # Update the frame
        self.update()
        self.frame2.update_idletasks()

    def change_category(self, *args):
        """
        Change the selected category
        """
        self.category = self.selectedCategory.get()

    def change_license(self, *args):
        """
        Change the seleted license
        """
        self.license = self.selectedLicense.get()

    def create_folders(self):
        """
        Create the category folders to deposit the images into
        """
        categories = self.categories # The labels/categories
        for category in categories:
            # Create the category and its parents if needed
            pathlib.Path("./categorized/" + category).mkdir(parents=True, exist_ok=True)

    def copy_to_category(self):
        """
        Copies the classified image to the chosen category folder

        Args:
            category: The name of the category to move the images to
        """
        # Check if user specified a new category
        if (self.newCategoryVar.get() != "None of the above. Enter new category name:" and self.newCategoryVar.get() != ""):
            newCat = re.sub('\W+', '', self.newCategoryVar.get().lower()) # Remove non word characters and convert to lowercase
            self.add_category(newCat)
            self.category = newCat

        # Find all the files in the current image's directory
        path = pathlib.Path("./TestImages/")
        files = [f for f in path.glob('**/{}/[!.]*'.format(self.shoot_name))]

        # Move them to the appropriate folder
        dst = "./categorized/{}/{}".format(self.category, self.shoot_name)
        pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
        for src in files:
            new_filename = self.slugify(re.match(r'.*/(.*)/(.*)$', str(src))[2]) # Ensure all file names are readable by browser
            copyfile(src, "{}/{}".format(dst, new_filename))

        # Write the metadata file
        self.write_metadata(dst)

        # Move on to the next image
        self.next_image()

    def add_category(self, category):
        """
        Add a new category to our category list and create it's folder

        Args:
            category: The name (string) for our new category
        """
        with open("./categories.txt", 'a') as categories:
            categories.write("{}\n".format(category))
        self.categories = load_labels("./categories.txt")
        self.create_folders()


    def write_metadata(self, dst):
        """
        Write the current image's metadata to 'metadata.json' with corresponding image files
        """

        metadata = {'name': '', 'date': '', 'credit': '', 'collection': '', 'tags': '', 'license': ''}

        if (self.creativeNameVar.get() != "Creative Name:" and self.creativeNameVar.get() != ""):
            metadata['name'] = self.creativeNameVar.get()

        if (self.dateVar.get() != "Date Collected: (Format like yyyy-mm-dd)" and self.dateVar.get() != ""):
            metadata['date'] = self.dateVar.get()

        if (self.creditVar.get() != "Photo Credit:" and self.creditVar.get() != ""):
            metadata['credit'] = self.creditVar.get()

        if (self.collectionVar.get() != "Collection:" and self.collectionVar.get() != ""):
            metadata['collection'] = self.collectionVar.get()

        if (self.tagsVar.get() != "Tags. Comma separated. E.g., tag1, tag2, tag3" and self.tagsVar.get() != ""):
            metadata['tags'] = re.split(r',\s*', self.tagsVar.get()) # Split on comma with or without spaces

        metadata['license'] = self.selectedLicense.get()

        with open("{}/metadata.json".format(dst), 'w') as outfile:
            json.dump(metadata, outfile)

    def slugify(self, string):
        """
        Ensure filenames can be read in a browser

        Args:
            string: The filename string to slugify

        Returns:
            The slugified string
        """
        string = unidecode.unidecode(string).lower()
        return re.sub(r'[^\w\.]+', '-', string)

if __name__ == "__main__":
    root = tk.Tk()
    classifier = ImageClassifier(root)
    tk.mainloop()
