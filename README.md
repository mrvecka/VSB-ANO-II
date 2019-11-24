# Checking occupacy of parking lot

Our job is to determine if there is a free parking spot on parking lot or not. We get the dataset for this purpose from one parking lot in Ostrava. The photos of parking lot has been taken in diferent day time and diferent year time. On each picture there is 56 parking spots.To exctract them we use information from **parking_map.txt**.

We introduce 2 aproaches how the goal can be achieved.

## Approach 1 - Sobel operator

We use sobel operator with kernel (3,3) to exctract vertical lines, then we use binary treshold. Count of non-zero pixels from result is our edge between occupied or free spot. Separately save counts for occupied and free spots. When all parking spots were processed, take lowest value from occupied spots and the heighes from free spots and make an average. The result is our final edge which will be used in test phase to determine if parking spot is occupied or not.

### Prerequisites
https://sourceforge.net/projects/opencvlibrary/
Download OpenCV from [OpenCV library](https://sourceforge.net/projects/opencvlibrary/) and include libraries to project

Unzip the dataset.zip

### Results

**Accuracy:** 93.37%

![example 1](/images/logo.png)
![example 2](/images/logo.png)

## Approach 2 - Classification using CNN and Tensorflow

We want to classify if parking spot is occupied or not, this is like made for NN classification. The base of second solution is convolutional neural network designed in Tensorflow framework with OpenCV for image pre-process. On input there is 1(grayscale) or 3(RGB) channels input image. We designed following NN:

![network architecture](/images/logo.png)

We use softmax_cross_entropy_with_logits to get the network error and AdamOptimizer for gradient distribution with **0.001** learning rate.

### Prerequisites
Tensorflow - 1.14.0
Tensorflow GPU -1.12.0
Python OpenCV - 3.4.3.18
Numpy = 1.17.3

Unzip the dataset.zip

### Results

**Accuracy:** 99.25%

![example 1](/images/logo.png)
![example 2](/images/logo.png)


