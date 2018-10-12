# Graphical Image Categorizer

**Setup:**
* Python 3.5
* Tensorflow
* Tkinter
* Numpy
* Unidecode

# To run:
```python3 main.py```

### How it works:

The application selects all JPEG images in the given directory. It then classifies the images, one at a time, with the trained *OW Image Classifier*. Each image has the option to add metadata which is written to .json files next to the images once categorized. Categorized images will be moved to a **categories** directory that contains subdirectories based on the labels fed into the classifier during training. Upon exiting and rerunning the application we first check the **categories** folder to skip the already processed images.
