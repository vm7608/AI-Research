# **Cycle Generative Adversarial Network (CycleGAN)**

## **1. Introduction**

Image-to-image translation is the task of transforming an image from one domain (e.g., images of zebras), to another (e.g., images of horses). Ideally, other features of the image — anything not directly related to either domain, such as the background — should stay recognizably the same

<p align="center">
  <img src="https://hardikbansal.github.io/CycleGANBlog/images/gan_example.jpg" >
  <br>
  <i>Example of converting images from one domain to another.</i>
</p>

Many previous works have applied deep learning to this problem by formulating it as a supervised learning problem on a dataset of training image pairs i.e the images of source and target domain should be of same location, and number of images of both the domains should also be same. However, obtaining paired examples isn’t always feasible.

<p align="center">
  <img src="https://lh6.googleusercontent.com/TISEaPBAOke5DtV_wBOi4bdXDHhvedvEcCOoldV45I9owIODfGO-GeZbPASenWETAfwE1-Ii0e8AO-Hz1yc6aiKaW821jqWukcu3ZqkZTFCRDW7P7zIxiKtbzVy4YF2X6-pj5iry1iR5khjTewJES3d2EJSxaySi5P55E6UkNIMJy1jel5IYQtvAKU4Dhw" >
  <br>
  <i>Example of paired and unpaired data.</i>
</p>

CycleGAN was introduced in Berkeley's famous 2017 paper, titled “Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks”. CycleGAN is an approach to training a deep convolutional neural network for image-to-image translation tasks. The Network learns mapping between input and output images using unpaired dataset.

The power of CycleGAN is able to learn transformations without one-to-one mapping between training data in source and target domains. It was interesting because it did not require paired training data — while an x and y set of images are still required, they do not need to directly correspond to each other. In other words, if you wanted to translate between sketches and photos, you still need to train on a bunch of sketches and a bunch of photos, but the sketches would not need to be of the exact photos in your dataset.

## **2. Architecture of CycleGANs**

### **2.1. Intuition of how CycleGANs work**

CycleGAN is a Generative Adversarial Network (GAN) that uses two generators and two discriminators.

#### **2.1.1. Example: Horses to Zebras**

Consider the problem where we are interested in translating images from horses to zebras and zebras to horses. We have 2 collections of photographs and they are unpaired (photos of different locations at different times; we don’t have the exact same scenes in zebras and horses)

- Collection 1: Photos of horses.
- Collection 2: Photos of zebras.

The intution is to develop an architecture of 2 GANs, and each GAN has a discriminator and a generator model.

- GAN A: Translates photos of horses (collection 1) to zebras (collection 2).
- GAN B: Translates photos of zebras (collection 2) to horses (collection 1).

Each GAN has:

- A *conditional generator model* that will synthesize an image given an input image.
- A *discriminator model* to predict how likely the generated image is to have come from the target image collection.

| Model | Generator Input | Generator Output | Discriminator Input | Discriminator Output |
|-|-|-|-|-|
| GAN A | Photos of horses | Photos of zebras | Photos of zebras from collection 2 and output from Generator A | Likelihood image is from collection 2 |
| GAN B | Photos of zebras | Photos of horses | Photos of horses from collection 1 and output from Generator B | Likelihood image is from collection 1 |

<p align="center">
  <img src="https://hardikbansal.github.io/CycleGANBlog/images/model.jpg" >
  <img src="https://hardikbansal.github.io/CycleGANBlog/images/model1.jpg" >
  <br>
  <i>Detail of CycleGANs Flow</i>
</p>

#### **2.1.2. Cycle Consistency**

So far, the models are sufficient for generating plausible images in the target domain but are not translations of the input image. So, each of the GANs are also updated using `cycle consistency loss`. This is designed to encourage the synthesized images in the target domain that are translations of the input image.

Cycle consistency loss compares an input photo to the generated photo and calculates the difference between the two, e.g. using the L1 norm or summed absolute difference in pixel values.

The first GAN (GAN A) will take an image of a horse, generate image of a zebra, this image is provided as input to the second GAN (GAN B), which in turn will generate an image of a horse.The cycle consistency loss calculates the difference between the image input to GAN A and the image output by GAN B and the generator models are updated accordingly to reduce the difference in the images.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*pLvOZXuDoyzh8BvinGW4Aw.png" >
  <br>
  <i>Example of Cycle consistency loss.</i>
</p>

### **2.2. Objective function**

There are 2 components to the CycleGAN objective function: an adversarial loss and a cycle consistency loss. Both are essential to getting good results.

- The adversarial loss is identical to the one used in the original GAN paper. It is a binary cross-entropy loss that encourages our generator to produce images that are indistinguishable from real images by our discriminator.

- The cycle consistency loss is a regularization term that encourages our generator to produce images that are close to the original input image. It can be the L1 distance between the input image and the reconstructed image.

Assume we have 2 GANs, G and F and 2 domains X and Y as following:

- $`G: X \to Y`$
- $`F: Y \to X`$
- $`D_y`$: Distinguisher y from G(x)
- $`D_x`$: Distinguisher x from F(y)

<p align="center">
  <img src="https://media.geeksforgeeks.org/wp-content/uploads/20200529210740/cycleconsistencyandlosses.PNG" >
  <br>
  <i>CycleGANs architecture.</i>
</p>

#### **2.2.1. Adversarial Loss**

Adversarial loss function is totally the same as GANs formulation. The aim of adversarial loss is to try simultaneously support generator generate fake image more realistic and increase discrimination capability between real and fake of discriminator.

There are two process of training in the cycleGAN. Thus, there are also exist two adversarial loss functions corresponding with each of them.

- Start from $`x`$, we try to generate fake image $`\hat{y}=G(x)`$:

```math
\mathcal{L}_{GAN}(G, D_{Y} , X, Y ) = \mathbb{E}_{y\sim p_{data}(y)}
[\log D_{Y} (y)]
+ \mathbb{E}_{x \sim p_{data}(x)}
[\log(1 − D_{Y} (G(x)))]
```

- And in the revert direction, start from $`y`$, we try to generate fake image $`\hat{x}=F(y)`$:

```math
\mathcal{L}_{GAN}(F, D_{X} , Y, X ) = \mathbb{E}_{x\sim p_{data}(x)}
[\log D_{X} (x)]
+ \mathbb{E}_{y \sim p_{data}(y)}
[\log(1 − D_{X} (F(y)))]
```

In the above equations, 2 generators $`G`$ and $`F`$ are trained to minimize the adversarial loss while 2 discriminators $`D_{X}`$ and $`D_{Y}`$ are trained to maximize the adversarial loss.

#### **2.2.2. Cycle Consistency Loss**

In the above image, we have 2 below process:

- Start from $`x`$:

```math
x \mapsto G(x) \mapsto F(G(x)) \approx x
```

- Start from $`y`$:

```math
y \mapsto F(y) \mapsto G(F(y)) \approx y
```

Cycle consistency loss function try to constrain both of these 2 process to generate output become as real as possible. It corresponds with minimizing the difference between $`F(G(x))-y`$ and $`G(F(y))-x`$ based on L1-Norm as following:

```math
\mathcal{L}_{cyc(G, F)} = \mathbb{E}_{x\sim p_{data}(x)}
[||F(G(x)) − x||_1] + \mathbb{E}_{y\sim p_{data}(y)}
[||G(F(y)) − y||_1]
```

The cycle in cycle consistency loss meaning that the learning process repeats in both direction.

#### **2.2.3. Full Objective**

Finally, we got the full objective function is summation of cycle consistency loss and adversarial loss:

```math
\mathcal{L}(G, F, D_X, D_Y ) =\mathcal{L}_{GAN}(G, D_X , X, Y )
+ \mathcal{L}_{GAN}(F, D_Y, X, Y)
+ \lambda\mathcal{L}_{cyc}(G, F)
```

$`\lambda`$ is a hyperparameter to balance between adversarial loss and cycle consistency loss.

### **2.3. Generator**

Each CycleGAN generator has three sections: an encoder, a transformer, and a decoder. The input image is fed directly into the encoder, which shrinks the representation size while increasing the number of channels. The encoder is composed of three convolution layers. The resulting activation is then passed to the transformer, a series of six residual blocks. It is then expanded again by the decoder, which uses two transpose convolutions to enlarge the representation size, and one output layer to produce the final image in RGB.

You can see the details in the figure below. Please note that each layer is followed by an instance normalization and a ReLU layer, but these have been omitted for simplicity.

<p align="center">
  <img src="https://images.viblo.asia/21fab617-eee8-41e9-bf74-7d31495db355.png" >
  <br>
  <i>Generator architecture of CycleGANs.</i>
</p>

As you can see above, the representation size shrinks in the encoder phase, stays constant in the transformer phase, and expands again in the decoder phase.

The representation size that each layer outputs is listed below it, in terms of the input image size, k. On each layer is listed the number of filters, the size of those filters, and the stride. Each layer is followed by an instance normalization and ReLU activation.

Since the generators’ architecture is fully convolutional, they can handle arbitrarily large input once trained.

### **2.4. Discriminator**

The discriminators are PatchGANs, fully CNN that look at a “patch” of the input image, and output the probability of the patch being “real”. This is both more computationally efficient than trying to look at the entire input image, and is also more effective — it allows the discriminator to focus on more surface-level features, like texture, which is often the sort of thing being changed in an image translation task.

<p align="center">
  <img src="https://images.viblo.asia/1696ebe2-b162-41a8-8f0b-92fc8bc88fdf.png" >
  <br>
  <i>An example architecture for a PatchGAN discriminator.</i>
</p>

PatchGAN is a fully convolutional network, that takes in an image, and produces a matrix of probabilities, each referring to the probability of the corresponding “patch” of the image being “real” (as opposed to generated). The representation size that each layer outputs is listed below it, in terms of the input image size, k. On each layer is listed the number of filters, the size of those filters, and the stride.

As you can see in the example architecture above, the PatchGAN halves the representation size and doubles the number of channels until the desired output size is reached.

In discriminator, the author use 70x70 patchGAN architecture to classify on each small patch, the combination of whole patches make the final result of discriminator more trusted. Such patch-level discriminator architecture also has fewer parameters than full-image discriminator.

<p align="center">
  <img src="https://phamdinhkhanh.github.io/assets/images/20201113_Pix2Pix/pic5.png" >
  <br>
  <i>PatchGAN architecture, the classification result is on each patch as on the figure</i>
</p>

## **3. Training CycleGANs**

To training process stable, authors replace the negative log likelihood function of Adversarial Loss by least-square loss as below:

```math
\mathbb{E}_{y \sim p_{\text{data}}(y) }[\log D_{Y} (y)] \to \mathbb{E}_{y \sim p_{\text{data}}(y)}[(D_{Y} (y))^2]
```

```math
\mathbb{E}_{x \sim p_{data}(x)}[\log(1 − D_{Y} (G(x)))] \to \mathbb{E}_{x \sim p_{data}(x)}[(1 − D_{Y} (G(x)))^2]
```

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*o7AV9LF_pdB9JdfVZlpmAQ.png" >
  <br>
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*uQmCNGYsJsvc9n9d-VqBjg.png" >
  <br>
  <i>Least-square Adversarial Loss</i>
</p>

In emperical, this way is more efficient when generate image with high quality.

<p align="center">
  <img src="https://av-eks-lekhak.s3.amazonaws.com/media/__sized__/article_images/Google_ChromeScreenSnapz098-thumbnail_webp-600x300.webp" >
  <br>
  <i>Example of Least Square Loss.</i>
</p>

The training stategy is process of found the optimization:

```math
G^{∗}, F^{∗} = \arg \min_{G,F} \max_{
D_X,D_Y} \mathcal{L}(G, F, D_X, D_Y )
```

- Maximization is to reinforce the strength of discriminator $`D_{X}`$ and $`D_{Y}`$ to distinguish real and fake image.
- Minimization is to train generator $`G`$ and $`F`$ to generate fake image more realistic and close to real image.

Actually, we can consider model are learning autoencoder $`G \circ F`$ and $`F \circ G`$ jointly. One image is mapped to it-self via an intermediate representation similiar to bottle neck layer function in autoencoder.

Some training details:

- $`\lambda`$ is set to 10 in all experiments.

- To reduce model oscillation, the descriminator is updated base on buffer of 50 history images instead of only lastest generator image. This help prevent the model from changing drastically from iteration to iteration.

- The Adam optimizer, a common variant of gradient descent, was used to make training more stable and efficient.

- The learning rate was set to 0.0002 for the first half of training (first 100 epochs), and then linearly reduced to zero over the remaining iterations (the next 100 epochs).

- The batch size was set to 1, which is why we refer to instance normalization, rather than batch normalization, in the architecture diagrams above.

## **4. Strengths and Limitations of CycleGANs**

Overall, the results produced by CycleGAN are very good — image quality approaches that of paired image-to-image translation on many tasks. This is impressive, because paired translation tasks are a form of fully supervised learning, and this is not.

<p align="center">
  <img src="https://images.viblo.asia/a7a99d48-7b3c-4d95-a0ae-a26da4c6165f.jpeg" >
  <br>
  <i>CycleGAN can be used for collection style transfer, where the entire works of an artist are used to train the model.</i>
</p>

CycleGAN works well on tasks that involve color or texture changes, like day-to-night photo translations, or photo-to-painting tasks like collection style transfer (see above). However, tasks that require substantial geometric changes to the image, such as cat-to-dog translations, usually fail.

<p align="center">
  <img src="https://i0.wp.com/nttuan8.com/wp-content/uploads/2020/05/6-2.png?resize=768%2C380&ssl=1" >
  <br>
  <i>A very unimpressive attempt at a cat-to-dog image translation.</i>
</p>

CycleGANs also have a tendency to create confusion about objects or to transform objects into the wrong thing. This is because the model is trained on unpaired data, and so it has no way of knowing what objects are supposed to be in the image. It can only learn to translate textures and colors.

<p align="center">
  <img src="https://i0.wp.com/nttuan8.com/wp-content/uploads/2020/05/failure_putin.jpg?resize=768%2C576&ssl=1" >
  <br>
  <i>CycleGAN detect mistakes surrounding objects as horses and then switches characteristics leading to confusion</i>
</p>
