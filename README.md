# A Scalable Template For Keras Deep Learning Projects

This project serves as a scalable template for deep learning projects built using Keras. 

Deep learning projects tend to have very similar building blocks. These blocks can be reusable but have to calibrated for each and every project. 

The essential pillars in a deep learning project, and thus this template are outlined below.

## Data Generators
When the amount of data is scaled up, and instead of 1000 images you are now working with 100,000 images you must seriously consider how data is loaded and processed for training. 

#### You have unlimited resources. Nice!
If you have access to a computer with a multi-terabyte RAM go for the simplest approach, just load the data in memory and train. 
#### Your resources are limited.
If you have a limitation in resources, the best approach is to load data part by part, even batch by batch. Data generators are used to generate train and test data on the fly. In other words, the data is loaded and pre-processed during training.


This template divides data generators into a reader and a batch generator component. A `DataReader` is a meta object that holds information about your data. Structure it however you like. The code resides in `reader.py`. 

### The `DataReader` Class
```python
# Example usage 
from PIL import Image
from reader import DataReader

# Define reader 
data_reader = DataReader()

# Depending on your implementation,
# you can iterate over image paths. 
for path = data_reader.image_paths:
	img = Image.open(path) # read using Pillow
	# Do whatever you like with the image
	img.flip()
```

You could have different `DataReader` objects for train, validation, and test sets. Or, just one, and divide your dataset using random indices assigned to train, validation and test sets. Up to you.  


### The `BatchGenerator` Class

The `BatchGenertor` class extends the `keras.utils.Sequence` class to enable proper integration with the Keras engine. You can find the code in `datagen.py`. Implement the places highlighted in `TODO`. These are:

- Add necessary parameters to the constructor, the `__init__` method.
- In the constructor, find the `self.index` variable. You must create a proper index to begin iterating through the data. The following example creates a mode aware index.  The mode defines whether a generator is for testing or training. 
```python
import random

import config # config holds global variables
# Must set random seed for deterministic division
# of train and test sets. 
random.seed(config.RANDOM_SEED_DEFAULT)
class BatchGenerator:
	# Example constructor for Batch Generator class
	def __init__(self, data_reader, mode="train" **kwargs):
		self.data_reader = data_reader
		self.mode = mode
		# Define index as a list of indices that correspond to
		# image data in the data_reader
		self.index = list(range(len(self.data_reader.image_data)))
		# Shuffle the index for random division based on mode
		random.shuffle(self.index)		
		# Filter index based on mode
		if self.mode == 'train' :
			self.index = self.index[:int(len(self.index) * config.TRAIN_RATIO_DEFAULT)]
		else:
			self.index = self.index[int(len(self.index) * config.TRAIN_RATIO_DEFAULT):]

``` 
  - A generic augmentation pipeline is already in place. Modify it to fit your needs.  The `imgaug` library is used for image augmentation. I find it to be comprehensive and well documented. The repo is [here](https://github.com/aleju/imgaug). 
  - The `__getitem__` method is for fetching a batch of data and labels. Depending on your project, your labels will differ in structure and format. Implement for your use case; load image and annotations to memory, preprocess the data, then append to the current batch.

The Batch Generator is done. If you would like to use the deployment functionalities in this repo, you must also complete the `ImageGenerator` class, which is just a simplified `BatchGenerator` that continuously generates images, without augmentation. This generator is used during the post-quantization stage.

## Models
Models are defined in separate files to impose reusability. Each file consists of a `construct` function that returns a Keras model. An example for MobileNet V1 is provided in `models/mobilenet_v1.py`.

One you define a model, go ahead to the config and add the `construct` method to the `MODELS` dictionary. Assign it a short and memorable key as you will be using this key to launch training scripts.

## Config
The config file is where all global and useful variables are kept. Some variables hold default values, others hold important information such as model constructors and path to the data directory.

Change the `DATA_DIR` to match with the path to your data Use this to read your data.

The configuration is important when restoring trained models. That's why a `config.json` file is generated every time a new model is trained. The `config.py` file contains a `read` and `save` method to read and save  configurations.

## Training

The `TODO` blocks in `train.py` should guide you through the necessary adjustments to get going.

**Data Preparation**
Adjust the inputs for`DataReader` and `BatchGenerator` according to your implementation. 

**Loss**
If you're using a generic loss function such as a Mean Squared Error, or a Categorical Cross Entropy, use the Keras implementations when compiling your model before training. Check out the Keras documentation for the proper names of these loss functions.

If using a custom loss, follow guidelines outlined by the Keras team. Implement your loss function in `loss.py` and pass it on as the loss parameter during model compilation. 

You might need loss weights for multi-task learning. You can pass weights as flags, however you must configure them in the compile function.

**Callbacks**
Callbacks are essential to track the progress of training. A callback is called at the end of each epoch. Check the `callbacks.py` file for callback options. The simple `cb = callbacks.get(**options)` one liner will get you going. Check the `callbacks.get` function defined in `callbacks.py`  to have an understanding of the options. 4 options are provided. These are for saving checkpoints, tensorboard monitoring, limiting the number of checkpoints and early stopping. 

**Train**
Use the following command to start training from the console.

```bash
python train.py --save_to progress/test --num_epochs 10 --batch_size 8 --model mobilenet_v2 --input_size 224,224 --jitter --overwrite
``` 

> The `jitter` option signals whether data augmentation should be applied

Use the following command to see your options for `train.py`.
```bash
python train.py --help
```

## Deploy 

The `deploy.py` script deploys trained models in various formats. These formats are the **Plain Keras**, **TensorFlow Lite**, **Edge TPU Compiled**. The idea is to simplify the process of generating ready to use model files right after training. 

> The **Plain Keras** model is in Hdf5 format.
>  The **Tensorflow Lite** model is quantized.
>   The **Edge TPU Compiled** model is for Google Coral accelerators.

Use the deploy script with the following command in the console. Replace read_from with path to the folder holding model checkpoints, and save_to with the path you want to save the models. 
```bash
python deploy.py --read_from progress/test --save_to deploy/test
``` 

## Contact
Drop an email at molinas.yoel@gmail.com for questions and comments.
