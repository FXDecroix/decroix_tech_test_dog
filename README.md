# decroix_tech_test_dog
Tech Test for Image Intelligence : Image resolution influence on dog/no dog classification 


## Requirements
The file requirements.txt contains all external dependencies versions required to install this package. To install them all, run the following command into the decroix_tech_test_dog directory:

```
pip3 install -r requirements.txt
```

## Installation and test

### Installation
To install the package, just run the following command into the decroix_tech_test_dog directory:

```
python setup.py install
```

### Test
To test the installation, run the following command in the same directory:

```
python -m unittest
```
If the 2 tests are successful, you should then be able to import the module multires_classif in python:

```python
python
>>> import multires_classif
```
## Usage

To train 2 networks with input image sizes 256 and 512 respectively, for 100 epochs and with a batch size of 32, you can run the following command:

```
python -m multires_classif.train_multires <path/to/data> <path/to/output> -e 100 -b 32
```

The data directory must follow the following architecture:

```
data_directory/
 |
 +-- labels.csv
 |    
 +-- images/
 |  |  
 |  +-- image1.jpg
 |  +-- image2.jpg
 |  +-- ...
```
The history of the training and the new weigths of the network are saved in the ouput directory as a dictionnary (.p) and .h5 files respectively.

## Discussion

### Dataset split and pre-processing

I treated the problem as binary classification (dog/no dog) using ResNet50 model, pretrained on ImageNet, in Keras. The dataset was split, in the standard way, as training (80%) and validation (20%). No cross-validation was performed, as the data seemed big enough.  

The input images are first zero-centered on each channel using the preprocess_input() function in Keras, relative to ResNet50, to have similar ranges, and prevent gradient to go out of control during back_propagation.  

I performed data_augmentation, by flipping vertically and horizontally, shifting, zooming and rotating. I kept this values quite low, because the ROI containing the dog can be close an adge of the image. Other transformations can be investigated.

### Network architecture

We first exclude the fully-connected layer on top of the network, to be able to input different image sizes. To fine tune the model, we replace the last fully-connected layers (1000 neurons) by a fully-connected layer with one neuron, and a sigmoid activation function, as we perform binary classification. Freezing all the previous layers leads to poor results, so for now they are left unfrozen.

### Training parameters

A batch size of 32 was initially chosen, but led to an Out Of Memory exception when running on the AWS instance (~11Go RAM) when feeding the network with images of 512. It had to be lowered at 8. The number of epochs is empirically chosen.   


## ToDo and further investigations

### ToDo

Write a documentation using Sphinx.

### Further investigations

#### Adjust parameters

Besides considering freezing layers, investing on dropout could be interesting in case of overfitting, as well as adding more fully-connected layers before last one.

#### Weakly supervised localization

The main issue in this approach is the noise in the image. In the training phase only a small part of the image contain the right information. However, the bounding boxes of these ROIs are not known. Some [recent works](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Oquab_Is_Object_Localization_2015_CVPR_paper.pdf) on weakly supervised object localization could then be a efficient approach to the model, while giving a localization of the object via Class Activation Maps.

 


