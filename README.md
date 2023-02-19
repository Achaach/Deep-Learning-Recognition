# Deep-Learning-Recognition

## Outline
In this project, we will use *convolutional neural nets* to classify images into different scenes.

Basic learning objectives:
1. Construct the fundamental pipeline for performing deep learning using PyTorch;
2. Understand the concepts behind different layers, optimizers.
3. Experiment with different models and observe the performance.

## Dataset
The dataset is in the ```data``` folder. It has two subfolders: ```train``` and ```test```. Go through any of the folder there are the folders with scene names like *bedroom*, *forest*, *office*. These are the 15 scenes that we want our model to predict given an image. All this data is labelled data provided to you for training and testing your model.

## Dataloader
The classes and the number of instances in the dataset:

Classes: {'forest': 0, 'bedroom': 1, 'office': 2, 'highway': 3, 'coast': 4, 'insidecity': 5, 'tallbuilding': 6, 'industrial': 7, 'street': 8, 'livingroom': 9, 'suburb': 10, 'mountain': 11, 'kitchen': 12, 'opencountry': 13, 'store': 14}

Train: 2985 instances 

Test: 1500 instances

## Accuracy
### SimpleNet:
1. Final training accuracy value: 0.6647 
2. Final validation accuracy value: 0.4727
### AlexNet:
1. Final training accuracy value: 0.9554 
2. Final validation accuracy value: 0.8733
### SimpleNetDropout
1. Final training accuracy value: 0.6345 
2. Final validation accuracy value: 0.5313

## Quantization
1. Size comparison: 74.97%
2. Processing Time comparison: 62.73%
3. Accuracy comparison: 0.61%
