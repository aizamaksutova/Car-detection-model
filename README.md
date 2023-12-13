# Car-detection-model
Implementation of Car detection model. This project involves developing a neural network classifier aimed at differentiating between images of machines and background scenes. It serves as a practical exercise in training neural networks, handling image data, and implementing data augmentation techniques.

## How to train a simple binary neural network classifier for car detection?

In the [src/detection_and_metrics.py/get_cls_model](https://github.com/aizamaksutova/Car-detection-model/blob/main/src/detection_and_metrics.py) function you can see my implementation of the model architecture, which is able to input image_size and return the model which we will further train. It is a convolutional neural network with 2 convolution layers.

```
nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 22, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )
```

The [src/detection_and_metrics.py/fit_cls_model](https://github.com/aizamaksutova/Car-detection-model/blob/main/src/detection_and_metrics.py) function trains the model and returns a trained model with accurate weights.

To check whether the model is trained correctly and gives away the accuracy of the classifier > 90% run: 

```
python3 tests/test_first_model.py 
```

## Model loading
For the first classifier if you don't want to train the model by yourself, you can download the classifier straight away from google drive by running the following:

```
import gdown

# a checkpoint
id_model = "1KVzsRIl6LruRYiNmxnh1Q1S34sRbSqKN"
output_model = "classifier_model.pth"
gdown.download(id=id_model, output=output_model, quiet=False)
```

## Neural network detector

Building upon the previously trained binary neural network classifier, this section of the project focuses on developing a neural network detector. This detector operates using the sliding window method to generate a confidence map indicating the presence of objects in fixed regions of an image.

### Conversion to Full Convolutional Neural Network:

The classifier is transformed into a full convolutional neural network (CNN) by replacing its fully connected layers with convolutional layers.
This conversion enables the network to output spatially precise confidence maps for object detection.
#### Layer Transformations:

##### Flatten and Linear Layer Replacement:
For a sequence of Flatten and Linear layers, where the Flatten layer inputs a tensor of size H × W × C_in and outputs a column vector for the Linear layer with C_out output values, we replace these with a C_out convolution with a kernel size of H × W.

##### Single Linear Layer Replacement:
A single Linear layer, taking an input vector of C_in numbers and outputting a vector of C_out numbers, is substituted by a C_out convolution with a kernel size of 1 × 1.

#### After replacing all fully-connected layers with convolutional layers, we obtain a fully-convolutional network.

This shift from the binary classificator to the fully-convolutional neural network to detect cars is described in the [src/detection_and_metrics.py/get_detection_model](https://github.com/aizamaksutova/Car-detection-model/blob/main/src/detection_and_metrics.py) function



