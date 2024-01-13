# **Conditional GAN (cGAN)**

## **1. Problem of GANs/DCGANs**

The original GANs/DCGANs have the following main disadvantages:

- Lack of control/input: The original GANs/DCGANs generate images based only on random noise inputs. There is no control or condition given to the generator. So, we cannot generate images of a specific class.
- Limited applications: Without conditional inputs, they are only suitable for tasks like generating natural images from noise, not for applications like text-to-image generation which require control.
- Inability to target specific domains: GANs/DCGANs may generate blurry or unrealistic images as they don't receive any domain-specific guidance.

In 2014, Mehdi Mirza and Simon Osindero, published the "Conditional Generative Adversarial Nets " paper, in which the generator and discriminator of the original GAN model are conditioned during the training on external information.

Conditional GANs (cGANs) that will help to generate images of a specific class based on additional information (labels y). y can be any additional information such as class labels, image descriptions, etc. and it can be seem as a condition to generate images so we call it conditional GANs (cGANs).

<p align="center">
  <img src="https://learnopencv.com/wp-content/uploads/2021/07/Conditional-GAN-in-PyTorch-and-TensorFlow.jpeg" >
  <br>
  <i>Flow Diagram of GANs vs cGANs.</i>
</p>

The idea is straightforward. Both generator and discriminator are fed a class label y and conditioned on it, as shown in the above figures. All other components are exactly like a typical GANs.

## **2. cGANs architecture**

### **2.1. Generator**

The CGANs Generator’s model is similar to DCGAN Generator’s model except for the one-hot vector, which is used to condition Generator outputs.

<p align="center">
  <img src="https://phamdinhkhanh.github.io/assets/images/20200809_ConditionalGAN/pic1.jpg" >
  <br>
  <i>Example of a generator architecture in cGANs.</i>
</p>

Generator takes a random noise vector z and a label y as input and produces an image x. First, noise vector z is pass through a fully-connected layer and be reshaped into 3D tensor (for example 7x7x128 as above image).

As the same time. labels y are being embedded by one-hot encoding and then reshaped into 3D tensor (size 7x7x1 as above image).

Then, the noise vector z and the label y are concatenated on the channel dimension (size 7x7x129 as above image) and passed through a series of `ConvTranspose2d` layers to produce an image labeled y.

<p align="center">
  <img src="https://machinelearningmastery.com/wp-content/uploads/2019/05/Example-of-a-Conditional-Generator-and-a-Conditional-Discriminator-in-a-Conditional-Generative-Adversarial-Network-1024x887.png" >
  <br>
  <i>Structure of a simple conditional generator.</i>
</p>

### **2.2. Discriminator**

The CGAN Discriminator’s model is similar to DCGAN Discriminator’s model except for the one-hot vector, which is used to condition Discriminator outputs.

<p align="center">
  <img src="https://learnopencv.com/wp-content/uploads/2021/07/Discriminator-fed-with-fake-example-and-Discriminator-fed-with-real-example-1.jpg" >
  <br>
  <i>Discriminator fed with fake example and Discriminator fed with real example</i>
</p>

Discriminator is still a binary classification model with task is to classify whether an image is real or fake. The real image come from the dataset and the fake image come from the generator. The ratio of real and fake images is 1:1.

The Discriminator is fed both real and fake examples with labels. It learns to not just recognize real data from fake, but also zeroes onto matching pairs. A pair is matching when the image has a correct label assigned to it. The Discriminator finally outputs a probability indicating the input is real or fake.

Its goal is to learn to:

- Accept all real sample label pairs.
- Reject all fake sample label pairs (the sample matches the label ).
- Also, reject all fake samples if the corresponding labels do not match.

For example, the Discriminator should learn to reject:

- The pair in which the generated image is 1, but the label was 2, regardless of the example being real or fake. The reason being it does not match the given label.
- All image-label pairs in which the image is fake, even if the label matches the image.

## **2.3. Loss function**

The loss function of cGANs is the same as GANs with a little change of adding labels y.

```math
\min_{G} \max_{D} V(D, G) = \underbrace{\mathbb{E}_{x \sim p_{data}(x)} [\log D(x|y)]}_{\text{log-probability that D predict x is real}} + \underbrace{\mathbb{E}_{z \sim p_{z}(z)} [\log (1-D(G(z|y)))]}_{\text{log-probability D predicts G(z) is fake}}
```

## **3. Training cGANs**

The training process of cGANs is the same as GANs. It consists of two main steps: the generator step and the discriminator step.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*sbnwxkciqzRzc2Ou.jpg" >
  <br>
  <i>Discriminator’s training process</i>
</p>

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*YiI0loO_1jnrLNkL.jpg" >
  <br>
  <i>Generator’s training process</i>
</p>
