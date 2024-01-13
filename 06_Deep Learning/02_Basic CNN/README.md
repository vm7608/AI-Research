# **CONVOLUTIONAL NEURAL NETWORKS**

## **1. Introduction to Image Processing**

### **1.1. Image**

- An image can be represented as a matrix of pixels. The number of pixels in an image is called the resolution of the image.
- The number of bits used to represent a pixel is called the bit depth of the image.

### **1.2 RGB Color System**

- A color in RGB is represented by a tuple of three numbers. Each number represents the intensity of the color, red, green and blue. The range of each number is from 0 to 255. When the three channels are combined, they produce a color image. So we have more than 16 million colors.

![RGB](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/RGB.png?resize=300%2C286&ssl=1)

### **1.3 Color Image**

- Let's consider an 800 x 600 color image below:

![Color Image](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/mathematical-bridge.jpg?w=800&ssl=1)

- The image size mean that the image has 800 pixels in width and 600 pixels in height. So we can represent the image as a matrix:

```math
\begin{bmatrix}
    w_{0,0} & w_{0,1} & \dots & w_{0,799} \\
    w_{1,0} & w_{1,1} & \dots & w_{1,799} \\
    \vdots & \vdots & \ddots & \vdots \\
    w_{599,0} & w_{599,1} & \dots & w_{599,799} \\
\end{bmatrix}
```

- In which, each $`w_{ij}`$ is a pixel. However, to represent a color image, we need three channels (r, g, b). Asume $`w_{ij} = (r_ij, g_ij, b_ij)`$ is a pixels, we have:

```math
\begin{bmatrix}
    (r_{0,0}, g_{0,0}, b_{0,0}) & (r_{0,1}, g_{0,1}, b_{0,1}) & \dots & (r_{0,799}, g_{0,799}, b_{0,799}) \\
    (r_{1,0}, g_{1,0}, b_{1,0}) & (r_{1,1}, g_{1,1}, b_{1,1}) & \dots & (r_{1,799}, g_{1,799}, b_{1,799}) \\
    \vdots & \vdots & \ddots & \vdots \\
    (r_{599,0}, g_{599,0}, b_{599,0}) & (r_{599,1}, g_{599,1}, b_{599,1}) & \dots & (r_{599,799}, g_{599,799}, b_{599,799}) \\
\end{bmatrix}
```

- For convenience of storage and processing, we separate each value in each pixel into a separate matrix. So we have 3 matrices of 600x800 to represent a color image.

```math
\begin{bmatrix}
    r_{0,0} & r_{0,1} & \dots & r_{0,799} \\
    r_{1,0} & r_{1,1} & \dots & r_{1,799} \\
    \vdots & \vdots & \ddots & \vdots \\
    r_{599,0} & r_{599,1} & \dots & r_{599,799} \\
\end{bmatrix}

\begin{bmatrix}
    g_{0,0} & g_{0,1} & \dots & g_{0,799} \\
    g_{1,0} & g_{1,1} & \dots & g_{1,799} \\
    \vdots & \vdots & \ddots & \vdots \\
    g_{599,0} & g_{599,1} & \dots & g_{599,799} \\
\end{bmatrix}

\begin{bmatrix}
    b_{0,0} & b_{0,1} & \dots & b_{0,799} \\
    b_{1,0} & b_{1,1} & \dots & b_{1,799} \\
    \vdots & \vdots & \ddots & \vdots \\
    b_{599,0} & b_{599,1} & \dots & b_{599,799} \\
\end{bmatrix}
```

- Now, with each color image, we have 3 matrices of 600x800, we call them 3 channels of the image: channel red, channel green, channel blue.

### **1.4. Tensor**

- Data is presented in 1 dimension called a vector. Data is presented in 2 dimensions called a matrix.

- When data have more than 2 dimensions, we call it a tensor. For example if we put 3 matrices of 600x800 together, we have a tensor of 3x600x800. To present the a color image in computer, we use a 3D tensor.

![Tensor](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/tensor.jpg?w=638&ssl=1)

### **1.5. Gray Image**

![Gray Image](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/gray.jpg?w=800&ssl=1)

- A gray image has only one channel. So we can represent a gray image as one matrix of pixel. In that matrix, each pixel is a number from 0 to 255. 0 is black and 255 is white, the value of pixel darker when near 0 and brighter when near 255.

- We can convert a color image to a gray image by using `NTSC formula`:

```math
Y = 0.299 R + 0.587 G + 0.114 B
```

### **1.6. Kernel**

- Kernel is a matrix of numbers. It is also called a filter. It is used to extract features from an image by performing a convolution operation.

- A kernel is usually a small square matrix. For example, a kernel of 3x3, 5x5, 7x7. why The kernel is usually an odd number because it has a center pixel.

- Some common kernel:
  - Indentity kernel: does not change the input image.
  - Edge detection kernel: detect the edge of an image.
  - Sharpen kernel: sharpen the input image.
  - Box blur kernel: blur the input image.

![Kernel](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/purpose.png?w=670&ssl=1)

### **1.7. Convolution**

- Convolution is a mathematical operation on two functions (f and g) to produce a third function that expresses how the shape of one is modified by the other. The term convolution refers to both the result function and to the process of computing it. It is defined as the integral of the product of the two functions after one is reversed and shifted.

- We define a kernel W of size 3x3 as follows:

```math
W = \begin{bmatrix}
    1 & 0 & 1 \\
    0 & 1 & 0 \\
    1 & 0 & 1 \\
\end{bmatrix}
```

- Calculation symbols of convolution operation: $`⊗`$. For example, $`Y = X ⊗ W`$ means that we perform a convolution operation on the image X and the kernel W to produce a convolved feature Y.

- With each element $`x_{ij}`$ in matrix X, we take a matrix with the same size with kernel W (called matrix A), starting from the element $`x_{ij}`$ as the center of the matrix. Then we multiply each element of the matrix with the corresponding element of the kernel W. Finally, we sum all the results of the multiplication to get the value of the element $`y_{ij}`$ in the convolved feature Y.

![Convolution](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/c1.png?resize=768%2C356&ssl=1)

- For example at $`x_{22}`$ (the red cell), matrix A has the same size with kernel W and $`x_{22}`$ is the center of matrix A. So we have:

```math
y_{11} = sum(A ⊗ W) = x_{11} * w_{11} + x_{12} * w_{12} + x_{13} * w_{13} + x_{21} * w_{21} + x_{22} * w_{22} + x_{23} * w_{23} + x_{31} * w_{31} + x_{32} * w_{32} + x_{33} * w_{33} = 4
```

- We can see that, the pixel in the outer border like $`x_{11}`$ will be ignored because it is not the center of matrix A. So that, matrix Y has a smaller size than matrix X. Size of matrix Y is calculated as follows (m is the number of rows of matrix X, n is the number of columns of matrix X, k is the size of kernel W):

```math
(m - k + 1) * (n - k + 1)
```

![Convolution](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/c2.png?resize=300%2C269&ssl=1)

![Convolution](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/giphy.gif?resize=526%2C384&ssl=1)

### **1.8. Padding and Stride**

#### **Padding**

- If we want the size of matrix Y is the same as the size of matrix X, we can add some zeros around matrix X. This technique is called padding.

![Padding](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/c3.png?w=490&ssl=1)

- For example, in the above figure we have padding = 1 which means that we add 1 zeros vector around matrix X.

#### **Stride**

- As above, we sequentially execute the elements in the matrix X and perform a convolution operation with the kernel W. The step size of the convolution operation is called stride. Here, we have stride = 1.

- If we have stride = k (k>1), we will only perform convolution operation on the elements of matrix X with the step size is k ($`x_{1+i*k, 1+j*k}`$).

- An example of padding = 1 and stride = 2:

![Stride](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/c5.png?w=492&ssl=1)

- Stride is often used to reduce the size of the output matrix.

#### **General**

- In general, if we have matrix X with m rows and n columns, kernel W with k rows and k columns, padding p and stride s, the size of the output matrix Y is calculated as follows:

```math
(\frac{m - k + 2p + 1}{s}) * (\frac{n - k + 2p + 1}{s}) \\
\
\\
(\text{round by add s - 1 to numerator}) \\
\
\\
(\frac{m - k + 2p + 1 + s - 1}{s}) * (\frac{n - k + 2p + 1 + s -1}{s}) \\
\
\\
\rightarrow (\frac{m - k + 2p}{s} + 1) * (\frac{n - k + 2p}{s} + 1)
```

## **2. Convolutional Neural Networks Architecture**

### **2.1. Introduction**

- Convolutional Neural Networks (CNNs) are a class of deep neural networks, most commonly applied to analyzing visual imagery.

- CNN has the following layers:
  - Convolutional layer: extract features from the input image by sliding a convolution filter over the input image.
  - Pooling layer: reduce the spatial size of the convolved feature.
  - Fully connected layer: classify the objects based on the output from the previous layer.

### **2.2. Problem of Fully Connected Layer**

![FCN](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/nn-2.png?resize=768%2C631&ssl=1)

- In the model of Neural Network, we have each neuron in a fully connected layer is connected to all the neurons in the previous layer. For example we have a color image size 64x64x3, to present it in computer, we use a tensor of 3x64x64. We flatten the tensor to a vector of 12288x1 and feed it to the input layer. The input layer is fully connected to the first hidden layer. The first hidden layer is fully connected to the second hidden layer. The second hidden layer is fully connected to the output layer.

- The problem is:
  - Too many parameters: The number of parameters in the model is too large.
  - In image, the nearby pixel will have some connections with each other. However, in fully connected layer, each neuron is connected to all the neurons in the previous layer. So the model does not take advantage of the spatial structure of the image.

- So by applying convolutional into layer of Neural network, we can reduce the number of parameters and take advantage of the spatial structure of the image.

![CNN](https://media.licdn.com/dms/image/C4D12AQE2lv1zPZrxVQ/article-inline_image-shrink_1500_2232/0/1619972210366?e=1694649600&v=beta&t=XuqD7xnO9uGZWBlnmrUNQfKoiqLlo6FUs2vWEojDPjc)

### **2.2. Convolutional layer**

#### **The first convolutional layer**

- In the above example, we only consider gray image:

![Convolutional layer](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/giphy.gif?resize=474%2C345&ssl=1)

- However, when we have a color image, we have 3 channels: red, green and blue. So when we perform image as a 3D tensor, we also need to define a kernel is a tensor with size of k x k x 3. For example, a kernel of 3x3x3:

![Convolutional layer](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/kernel.png?w=326&ssl=1)

- We define kernel have the same depth with the input image. So that, we can perform convolution operation on each channel of the input image by moving kernel through the input image. The result of the convolution operation is a convolved feature.

![Convolutional layer](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/conv.png?resize=768%2C515&ssl=1)

![Convolutional layer](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/conv.gif?resize=1000%2C562&ssl=1)

- Comments: Ouput Y of convolution operation is a matrix.

- Because with each kernel, we can extract a feature from the input image. So that, we can use multiple kernels to extract multiple features from the input image. The result of the convolution operation is a tensor of convolved features. Ouput of a kernel is a matrix, so with k kernel we have k matrix. We stack k matrix together to have a tensor of convolved features with depth is k.

![Convolutional layer](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/conv1.png?w=855&ssl=1)

- The output of the first convolutional layer will be the input of the next layer.

#### **General Convolutional Layer**

- Asume that input of a General Convolutional Layer is a tensor of H x W x D, we have:
  - H: height of the input image.
  - W: width of the input image.
  - D: depth of the input image.

- We have a kernel of size F x F x D (kernel depth is equal to input depth and F is odd).
- We have padding P and stride S.

- Convolutional layer that applied K kernels, so the output of the convolutional layer is a tensor with size as follow:

![Convolutional layer](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/conv2-1.png?w=991&ssl=1)

- Note:
  - Output of convolutional layer will go through an activation function before being passed to the next layer.
  - The total parameters of a convolutional layer is calculated as follows: each kernel has F x F x D + 1 parameters (F x F x D parameters for weights and 1 parameter for bias). So with K kernels, we have K x (F x F x D + 1) parameters.

### **2.3. Pooling layer**

- Pooling layers are often used between convolutional layers, to reduce the size of the data while preserving important properties. The reduced data size reduces computation in the model.

- There are 2 common types of pooling layer:
  - Max pooling: take the maximum value in the pooling window.
  - Average pooling: take the average value in the pooling window.

![Pooling layer](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/pooling_2.jpg?w=596&ssl=1)

- Asume that pooling size is K x K, input of pooling layer is a tensor of H x W x D, we will:
  - Split tensor into D matrix with size of H x W.
  - In each matrix, on the area of ​​size K x K we find the maximum or average of the data and then write it in the resulting matrix.
  - The rules for stride and padding apply like convolution on images.

- An example of max pooling layer size 3x3, stride = 1 and padding = 0:

![Pooling layer](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/pooling.gif?resize=396%2C248&ssl=1)

- However, almost we use max pooling layer with size 2x2, stride = 2 and padding = 0, then output of pooling layer is a tensor of H/2 x W/2 x D. (height and width is reduced by half, depth is not changed). For example:

![Pooling layer](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/pooling.jpeg?w=514&ssl=1)

### **2.4. Fully connected layer**

- After the image is passed through many convolutional layers and pooling layers, the model has learned the relative features of the image (eg eyes, nose, face frame, ...) then the tensor of the output of the final layer, size H x W x D, will be converted to a dimension vector with size of H x W x D. Then we use fully connected layers to combine the features of the image to get the output of the model.

![Fully connected layer](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/flattern.png?resize=768%2C367&ssl=1)

- A fully example of a CNN model: Input image -> Convolutional layer (Conv) + Pooling layer (Pool) -> Fully connected layer (FC) -> Output.

![Visualizing CNN](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/cnn.png?w=1024&ssl=1)

- In the figure above, we have a CNN model with 2 convolutional layers, 2 pooling layers and 2 fully connected layer. `k` is the kernel size, `f` is number of filters, `s` is the stride.

---
The end.
