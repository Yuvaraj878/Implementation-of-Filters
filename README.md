# Implementation-of-Filters
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import the necessary modules.

### Step2
For performing smoothing operation on a image.

    Average filter
```python
kernel=np.ones((11,11),np.float32)/121
image3=cv2.filter2D(image2,-1,kernel)
```
    Weighted average filter
```python
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
```
    Gaussian Blur
```python
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
```
    Median filter
```python
median=cv2.medianBlur(image2,13)
```
### Step3
For performing sharpening on a image.

    Laplacian Kernel
```python
kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
```
    Laplacian Operator
```python
laplacian=cv2.Laplacian(image2,cv2.CV_64F)
```
### Step4
Display all the images with their respective filters.
## Program:
```PYTHON
### Developed By   : YUVARAJ.S
### Register Number: 22008589
import cv2
import matplotlib.pyplot as plt
import numpy as np
image1=cv2.imread("img.jpg")
image2=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
plt.imshow(image1)
plt.axis("off")
plt.show()
plt.imshow(image2)
plt.axis("off")
plt.show()
```
### 1. Smoothing Filters

i) Using Averaging Filter
```Python
### Developed By   : YUVARAJ.S
### Register Number: 22008589
kernel=np.ones((11,11),np.float32)/121
image3=cv2.filter2D(image2,-1,kernel)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Average Filter Image")
plt.axis("off")
plt.show()



```
ii) Using Weighted Averaging Filter
```Python
### Developed By   : YUVARAJ.S
### Register Number: 22008589
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()




```
iii) Using Gaussian Filter
```Python
### Developed By   : YUVARAJ.S
### Register Number: 22008589

gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()


```

iv) Using Median Filter
```Python

### Developed By   : YUVARAJ.S
### Register Number: 22008589
median=cv2.medianBlur(image2,13)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Median Blur")
plt.axis("off")
plt.show()



```

### 2. Sharpening Filters
i) Using Laplacian Kernal
```Python
### Developed By   : YUVARAJ.S
### Register Number: 22008589
kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")
plt.show()




```
ii) Using Laplacian Operator
```Python
### Developed By   : YUVARAJ.S
### Register Number: 22008589
laplacian=cv2.Laplacian(image2,cv2.CV_64F)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()




```

## OUTPUT:

### 1. Smoothing Filters

i) Using Averaging Filter
![](./2nd.png)
ii) Using Weighted Averaging Filter
![](./3ed.png)

iii) Using Gaussian Filter
![](./4th.png)

iv) Using Median Filter
![](./5th.png)

### 2. Sharpening Filters
i) Using Laplacian Kernal
![](./6th.png)
ii) Using Laplacian Operator
![](./7th.png)

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
