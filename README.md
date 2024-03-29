# Task Forest

Task forest is a project to help better define and select the tasks for inference in mobile agents.

## What is task?

Task in AI is defined as a tuple which combines the training set, model and the attribute that guide for the usage of the task. Note that task here is something like the model itself. We think that task can have a better structure, which will be shon as follow.

## What will we do?

We will first focus on the inference of yolov3 models for mobile agents such as smart cars. We will try to divide the whole dataset to multi tasks which will work better in different scenarios and then select the best one for inference. We think this way will get a set of model with better performance and smaller space.

We think there are two possible ways:

### Top down

All the samples can be seen as a task and the other tasks will be generated from the samples that work badly in the existing samples.

### Bottom up

Each sample can be seen as a task and then the final tasks will be generated from tasks merging.

## Quick Start

This project is built on mmDetection. The supported dataset is BDD and Cityscape. For the running of cityscape, please run the file: forest_train_cityscape.py; For the running of bdd, please run the file: forest_train_bdd.py.

## Related Publishment

Zheng Z, Li Y, Song H, et al. Towards Edge-Cloud Collaborative Machine Learning: A Quality-aware Task Partition Framework[C]//Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 2022: 3705-3714.
