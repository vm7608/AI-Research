# **Deep Convolutional Generative Adversarial Networks (DCGANs)**

## **1. Introduction**

In the original GAN paper, the authors used a multilayer perceptron (MLP) as the generator and the discriminator. However, MLPs are not suitable for image generation because they do not take into account the spatial structure of the image.

So, to generate photorealistic images, we need to use a convolutional neural network (CNN) instead of an MLP. This is where Deep Convolutional GANs (DCGANs) come in.

DCGANs were introduced in the paper "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" by Radford et al. in 2015.

The main change in DCGANs is the use of convolutional and convolutional-transpose layers instead of fully-connected layers.

- The generator uses `ConvTranspose2d` layers to produce an image from a random noise vector.
- The discriminator uses `Conv2d` layers to produce a single scalar output.

## **2. Architecture of DCGANs**

<p align="center">
  <img src="https://tjmachinelearning.com/lectures/1718/gan/dcgan.png" >
  <br>
  <i>Architecture of the generator and the discriminator in a DCGAN model.</i>
</p>

The following are the main architectural features of DCGANs.

### **2.1. Replace any pooling layer with convolutional stride**

In CNNs, max pooling is used to reduce the spatial dimensions of the feature maps. However, max pooling is not suitable for GANs because it discards information about the location of features in the feature maps.

So, DCGANs replace all max pooling layers with convolutional layers with stride. That will allow the network to learn its own spatial downsampling. This approach is used in Generator, allowing it to learn its own spatial upsampling, and also in Discriminator, allowing it to learn its own spatial downsampling.

### **2.2. Use transposed convolution for upsampling**

Transposed convolution is used in the generator to produce an image from a random noise vector.

Transposed convolution is also known as fractionally-strided convolution or deconvolution. It is the opposite of convolution. A transposed convolutional layer is usually carried out for upsampling. It takes a small input tensor/feature map and produces a larger output tensor/feature map.

If a convolution with stride > 1 will reduce the spatial dimensions of the feature maps, a transposed convolution with stride > 1 will increase the spatial dimensions of the feature maps.

Just like the standard convolutional layer, the transposed convolutional layer is also defined by the padding and stride.

<p align="center">
  <img src="https://i0.wp.com/nttuan8.com/wp-content/uploads/2020/04/t2.png?resize=1536%2C366&ssl=1" >
  <br>
  <i>Step by step of transposed convolution.</i>
</p>

Implementing a transposed convolutional layer can be explained in 4 main steps:

- Step 1: Calculate new parameters z and p’
- Step 2: Between each row and columns of the input, insert z number of zeros. This increases the size of the input to:
  
```math
(2 \times i-1) \times (2 \times i-1)
```

- Step 3: Pad the modified input image with p’ number of zeros
- Step 4: Carry out standard convolution on the image generated from step 3 with a stride length of 1

For a given size of the input (i), kernel (k), padding (p), and stride (s), the size of the output feature map (o) generated is given by:

```math
o = s \times (i-1) + k - 2p
```

For the example in above image, we have stride = 2, padding = 1, kernel = 3, input = 4, so:

```math
o = 2 \times (4-1) + 3 - 2 \times 1 = 7
```

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*zbVS6lHvo9J4aRZeE-77lA.png" >
  <br>
  <i>2 types of convolution.</i>
</p>

### **2.3. Remove fully connected hidden layers for deeper architectures**

The author of DCGANs noted that:

- The state of art trend towards eliminating fully connected layers on top of convolutional features. The strongest example of this is global average pooling which has been utilized in state of the art image classification models instead of fully connected layers. But the author found that global average pooling increased model stability but hurt convergence speed.

- A "middle ground" approach they used that worked well was to directly connect the output of the highest convolutional layer in the generator/discriminator to the input and output respectively, without going through any fully connected layers.

- In DCGANs, the first layer of the GAN, which takes a uniform noise distribution Z as input, could be called fully connected as it is just a matrix multiplication, but the result is reshaped into a 4-dimensional tensor and used as the start of the convolution stack. For the discriminator, the last convolution layer is flattened and then fed into a single sigmoid output.

### **2.4. Use Batch Normalization**

Batch normalization is a technique for improving the speed, performance, and stability of learning by normalizing the input to each unit to have zero mean and unit variance. This helps deal with training problems that arise due to poor initialization and helps gradient flow in deeper models.

This proved critical to get deep generators to begin learning, preventing the generator from collapsing all samples to a single point which is a common failure mode observed in GANs.

Directly applying batchnorm to all layers however, resulted in sample oscillation and model instability. This was avoided by not applying batchnorm to the generator output layer and the discriminator input layer.

<p align="center">
  <img src="https://i.stack.imgur.com/VEQhM.png" >
  <br>
  <i>Batch Normalization process</i>
</p>

### **2.5. Activation function**

The ReLU activation function is used in the generator for all layers except the output layer, which uses the tanh activation function. Tanh is used in the output layer of generator because the input images are normalized to [-1, 1].

This is due to the fact that when generating the images, we are typically normalized to be either in the range [0,1] or [-1,1]. So if we want output images to be in [0,1] we can use a sigmoid and if we want them to be in [-1,1] we can use tanh.

The author observed that using a bounded activation allowed the model to learn more quickly to saturate and cover the color space of the training distribution.

Within the discriminator, they also found that The LeakyReLU activation function work
well, especially for higher resolution modeling. So LeakyReLU is used in the discriminator for all layers.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53233519811_d71ea0f69b_o.png" >
  <br>
  <i>Some common activation functions</i>
</p>

### **2.6. Loss function**

The loss function used in DCGANs is the same as the loss function used in GANs.

```math
\min_{G} \max_{D} V(D, G) = \underbrace{\mathbb{E}_{x \sim p_{data}(x)} [\log D(x)]}_{\text{log-probability that D predict x is real}} + \underbrace{\mathbb{E}_{z \sim p_{z}(z)} [\log (1-D(G(z)))]}_{\text{log-probability D predicts G(z) is fake}}
```

## **3. Training DCGANs**

The training process of DCGANs is similar to that of GANs. It consists of two main steps: the generator step and the discriminator step.

DCGANs was trained on three datasets: Large-scale Scene Understanding (LSUN); Imagenet-1k; and a newly assembled Faces dataset. The author found that DCGANs can learn a hierarchy of representations from object parts to scenes in both LSUN and Imagenet-1k. They also found that DCGANs can learn an interesting representation of the facial features in the Faces dataset.

No pre-processing was applied to training images besides scaling to the range of the tanh activation function [-1, 1].

All models were trained with mini-batch stochastic gradient descent (SGD) with a mini-batch size of 128. All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02.

In the LeakyReLU, the slope of the leak was set to 0.2 in all models.

While previous GAN work has used momentum to accelerate training, the author used Adam optimizer with tuned hyperparameters. The author found the suggested learning rate of 0.001, to be too high, using 0.0002 instead. Additionally, the author found leaving the momentum term $\beta_1$ at the suggested value of 0.9 resulted in training oscillation and instability while reducing it to 0.5 helped stabilize training.
