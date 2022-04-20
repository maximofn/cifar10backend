# CIFAR10 backend 

Backend where a neural network trained in cifar10 is applied to an image and returns the class

If from a terminal you run `curl -X POST -F 'file=@<image path>' https://cifar10backend.herokuapp.com//densenet161` it will return the class and the confidence percentage