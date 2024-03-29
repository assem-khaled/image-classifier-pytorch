# Image Classifier project for Udacity Nanodegree program

This project is a part of Udacity's Intro to Machine learning with pytorch Nanodegree program. In this project, the goal is to develop code for 
an image classifier built with PyTorch, then convert it into a command line application.

This project uses PyTorch and the torchvision package; the Jupyter Notebook walks through the implementation of the 
image classifier and shows examples of the classifier's prediction on a test images. The classifier converted into a python 
application which could be used from command line using "train.py" and "predict.py".

The image classifier to recognize different species of flowers. [Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) 
contains 102 flower categories.

## Command line application
in the command line application there is an option to select a [VGG](https://pytorch.org/hub/pytorch_vision_vgg) architecture network.\
To use train.py to train a model; use the following positional and optional parameters:
- 'data_directory': Provide data directory
- '--model_save_dir': option to provide save directory for the model
- '--checkpoint': option to save a only checkpoint (default: False)
- '--arch': option to use vgg11, vgg13, vgg16 or vgg19 architecture (default: vgg11)
- '--learning_rate': option to provide learning rate (default: 0.001)
- '--hidden_units': option to provide number of hidden units (default: 512)
- '--epochs': option to provide number of epochs to train (default: 20)
- '--gpu': option to use gpu
- '--test': option to test the model on the test dataset

To use predict.py to make prediction; use the following positional and optional parameters:
- 'image': Provide image directory
- 'model': Provide model directory
- '--checkpoint': option to load a checkpoint (default: False)
- '--top_k': option to get top k number predictions (default: 5)
- '--category_names': option to provide another directory for the mapping category names (default: [cat_to_name.json](cat_to_name.json))
- '--gpu': option to use gpu
