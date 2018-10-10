# OW Image classification

**Setup:**
* Python 3.5
* Tensorflow
* CUDA 9.0
* CUDANN 7.0.5

# Data:

### Collect data:
* [Google Images Downloader](https://github.com/hardikvasa/google-images-download).It's fast, easy, simple and efficient.
* More data is better. Download lots

### Augmentation:
* The number you chose was probably still too low or Google didn't download all the images successfully. So we need to Augment the images to get more of them!
* You can use the following to do it easily, [Augmentor](https://github.com/mdbloice/Augmentor)

*Careful: While Augmenting, be careful about what kind of transformation you use. Some things aren't the same upside down and backwards!*

### Standardize:
* After Augmentation, make a folder named rawdata in the current working directory.
* Create folders with their respective class names and put all the images in their respective folders
* Run ```preprocess.py```
* This will resize all the images to a standard resolution and same format and put it in a new folder named data

### Training:
* Run ```python3 retrain.py --image_dir ./data``` and wait forever

### Running the model:
* Run ```python3 label_image.py --graph=./trainedModel/output_graph.pb --labels=./trainedModel/output_labels.txt --input_layer=Placeholder --output_layer=final_result --image={INPUT IMAGE TO TEST}```

### Tensorboard

#### View pretty graphs!

To start tensorboard run:

```tensorboard --logdir trainedModel/retrain_logs/```
