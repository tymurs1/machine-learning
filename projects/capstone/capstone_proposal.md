# Machine Learning Engineer Nanodegree
## Capstone Proposal
Tymur Salakhutdinov 
April 21th, 2023

## Proposal

### Domain Background

Factories and warehouses seek to automatize their processes in order to improve their productivity. With the rise of artificial intelligence and image recognition technologies companies enhance their facilities with the systems which help to optimize the storage space, spot errors in production, identify and sort objects. Amazon Fulfillment Centers utilize different innovations that allow it to deliver products all over the world with the help of robotic and computer vision technologies. These products are placed in bins randomly based on where some space is available. Thus each bin can contain different kind of products and different number of products. Sometimes the items get misplaced and the contents of some bin does not match with the recorded inventory of that bin. Ultimately the goal is to identify a number of items in a bin based on the picture of its contents.

This is a problem of visual recognition and it is usually solved by applying convolutional neural networks. Some research on object counting the number of items has been done in https://github.com/silverbottlep/abid_challenge. The paper http://cs229.stanford.edu/proj2018/report/65.pdf shows comparison of different models' performance  when identifying the number of objects in bins.

### Problem Statement

The problem to be solved is that occasionally the number of items in some bin does not match with the recorded inventory. That being said when the actual number of the items in the bin does not match the recorded number it means the problem has occurred. In order to prevent this error going forward it is required to build a model which will count the number of items in each bin.

### Datasets and Inputs

Working dataset contains 10,441 raw color photos of the products in the bins. The photos are sorted in folders "1", "2", "3", "4", "5" 
which corresponds to correct number of items in the bins. This is a subset of the full Bin Image Dataset which Amazon published in order to find the best solution to the inventory mismatch problem https://registry.opendata.aws/amazon-bin-imagery/ . The working dataset will be split into 3 subsets which will be used to train, validate and test a convolutional neural. Upon preliminary evaluation of the input dataset there have been found a number of photos with such a large noise and occlusions that it is not possible even for a human to count the number of items.

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

### Solution Statement

In order to solve the problem I will train and finetune already existing Convolutional Neural Network Resnet50 via transfer learning. I will measure the performance of the model using simple accuracy which is a ratio of a number of correctly recognized bins to the total number of bins.

### Benchmark Model

The benchmark model for this problem is the model built by Pablo Bertorello et al. and described in this paper -  http://cs229.stanford.edu/proj2018/report/65.pdf. Based on that research Convolutional Neural Networks performed the best comparing to other models - Logistic Regression, Classification Trees, Support Vector Machine. The main metric used in the research is validation accuracy. There is also training accuracy and Root Mean Squared Error metric considered in the research. The best accuracy on the different CNNs varied in the range of 50-56%. Noisy photos of the bins caused confounding the training dataset which resulted in lower accuracy. It is also stated in the paper that ensemble model usage might improve overall performance.

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics

I will use accuracy as an evaluation metric to quantify performance. In this case accuracy is calculated as a ratio of a number of correctly recognized bins to the total number of bins. This seems to be the most important metric for this task as we want to correctly recognize as many bins as possible. For this task we will have to maximize the accuracy. 

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design

First of all I will do some exploratory data analysis which includes checking sample images from different categories: with 1, 2, 3, 4 or 5 items in a bin. I will try to see if there are many images which are too noisy. I might consider manual data cleaning if there are too many noisy images. Then I will display mean images for every category and then I will do Principal Component Analysis (PCA) to visualize the components that describe each class the best. 
As the next step I will split the dataset of images into 3 subsets: training, validation and testing. Training set will be used to train the model, validation dataset will be used for validation of the accuracy and testing dataset will serve as a final evaluation of the model which will be created as a result of the whole training process. 
I will prepare a function which will normalize the images, and do a transformation of a random horizontal flip which is supposed to improve  learning of some features of the images and the whole training in general.

I will train Resnet50 model via transfer learning. I am considering one or two fully connected layers which I will tune during the training process. The pretrained weights will remain the same. I will also consider other Convolutional Neural Networks and an ensemble of CNNs. 

I will perform hyperparameter optimization in order to find the best hyperparameters such as batch size and learning rate. I will consider optimization methods of Stochastic Gradient Descent (SGD) and ADAM. The loss function will be Cross-Entropy Loss as it is quite common for the classification problems. For the training I will use PyTorch library and for the optimization I will utilize sagemaker library.

I will consider the accuracy over 50% a very good result as it will be comparable with the benchmark research accuracy. 

Once the model is trained I will deploy it to an endpoint and make requests with some sample photos. 

