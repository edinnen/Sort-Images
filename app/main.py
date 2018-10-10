import tkinter as tk
from PIL import Image, ImageTk
import os
import numpy as np
from label_image import *

class ImageClassifier(tk.Frame):

    def __init__(self, parent, *args, **kwargs):

        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.root = parent
        self.root.wm_title("Classify Image")
        src = "./TestImages/"

        self.list_images = []
        for d in os.listdir(src):
            self.list_images.append(d)

        self.frame1 = tk.Frame(self.root, width=500, height=400, bd=2)
        self.frame1.grid(row=1, column=0)
        self.frame2 = tk.Frame(self.root, width=500, height=400, bd=1)
        self.frame2.grid(row=1, column=1)

        self.cv1 = tk.Canvas(self.frame1, height=390, width=490, background="white", bd=1, relief=tk.RAISED)
        self.cv1.grid(row=1, column=0)
        self.cv2 = tk.Canvas(self.frame2, height=390, width=490, bd=2, relief=tk.SUNKEN)
        self.cv2.grid(row=1, column=0)

        claButton = tk.Button(self.root, text='Confirm', height=2, width=10, command=self.classify_obj)
        claButton.grid(row=0, column=1, padx=2, pady=2)
        nextButton = tk.Button(self.root, text='Next', height=2, width=8, command=self.next_image)
        nextButton.grid(row=0, column=0, padx=2, pady=2)

        self.counter = 0
        self.max_counter = len(self.list_images)-1
        self.next_image()

    def classify_obj(self):
        model_file = "./output_graph.pb"
        label_file = "./output_labels.txt"
        input_height = 299
        input_width = 299
        input_mean = 0
        input_std = 255
        input_layer = "Placeholder"
        output_layer = "model"

        graph = load_graph(model_file)
        t = read_tensor_from_image_file(
            "{}{}".format("./TestImages/", self.list_images[self.counter]),
            input_height,
            input_width,
            input_mean,
            input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        with tf.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)
        values = '';
        for i in top_k:
            # print(labels[i], results[i])
            values = values + str(labels[i]) + ' => ' + str(results[i]) + '\n';
        self.cv2.delete("all")
        self.cv2.create_text(125, 65, fill="darkblue", font="Roboto 15", text=values)

    def next_image(self):
        if self.counter > self.max_counter:
            self.cv1.delete("all")
            self.cv2.delete("all")
            self.cv1.create_text(125, 65, fill="darkblue", font="Roboto 15", text="No more images!")
        else:
            im = Image.open("{}{}".format("./TestImages/", self.list_images[self.counter]))
            if (490-im.size[0])<(390-im.size[1]):
                width = 490
                height = width*im.size[1]/im.size[0]
                self.next_step(height, width)
            else:
                height = 390
                width = height*im.size[0]/im.size[1]
                self.next_step(height, width)

    def next_step(self, height, width):
        self.im = Image.open("{}{}".format("./TestImages/", self.list_images[self.counter]))
        self.im.thumbnail((width, height), Image.ANTIALIAS)
        self.root.photo = ImageTk.PhotoImage(self.im)
        self.photo = ImageTk.PhotoImage(self.im)

        if self.counter == 0:
            self.cv1.create_image(0, 0, anchor = 'nw', image = self.photo)

        else:
            self.im.thumbnail((width, height), Image.ANTIALIAS)
            self.cv1.delete("all")
            self.cv1.create_image(0, 0, anchor = 'nw', image = self.photo)

        self.classify_obj()

        self.counter += 1

if __name__ == "__main__":
    root = tk.Tk()
    MyApp = ImageClassifier(root)
    tk.mainloop()
