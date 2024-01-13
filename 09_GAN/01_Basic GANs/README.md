# **Generative Adversarial Networks (GANs)**

## **1. Overview and introduction**

### **1.1. What is GANs?**

Generative Adversarial Networks (GANs) were developed in 2014 by Ian Goodfellow and his teammates. GAN is basically a generative model that generates a new set of data based on training data that look like training data.

To understand the term GAN let’s break it into separate three parts:

- Generative – A generative model, which can captures the distribution of training data and can be used to generate new samples.
- Adversarial – The training of the model is done in an adversarial setting.
- Networks – Use deep neural networks for training purposes.

GANs have two main blocks (two neural networks) which compete with each other. They are Generator and Discriminator. The generator tries to generate new data instances while the discriminator tries to distinguish between real and generated data.

### **1.2. Discriminative vs Generative models**

Machine learning models can be classified into two categories: discriminative and generative models (as one of many ways to classify ML models).

<p align="center">
  <img src="https://imgur.com/xjtCWqw.png" >
  <br>
  <i>Categories of Machine Learning</i>
</p>

`Discriminative models` base on input features x to predict label or value of output y (e.g. classification, regression). The predicted value is a conditional probability based on input features: `P(y|x)`. For example in binary classification:

```math
p(y|\mathbf{x}) = \frac{1}{1+e^{-\mathbf{w}^\intercal\mathbf{x}}}
```

Generative models, on the other hand, try to learn the probability distribution `P(x|y)` of the input features x and the label y. The models will concentrate on finding what is the input features properties when we already know the label y. Generative models usually based on `Bayes theorem`:

```math
P(x|y) = \frac{P(y|x)P(x)}{P(y)}
```

For example, we have a dataset of bad dept with 2 input features x1, x2 and 1 output label y (0: Non-Fraud, 1: Fraud).

<p align="center">
  <img src="https://imgur.com/dWC2zfO.png" >
  <br>
  <i>Example of discriminative and generative model</i>
</p>

- In the left, discriminative model will try to find the boundary between 2 classes (Non-Fraud and Fraud) based on input features x1, x2.
- In the right, generative model will try to find the probability distribution of input features x1, x2 when we already know the label y (Non-Fraud or Fraud). Based on that, it can generate new samples that look like the training data (blue square). Note that, with generative models, we need to know the label y of data.

### **1.3. Types of Generative models**

There are two types of generative models: explicit and implicit models.

- `Explicit model`: try to find probability distribution of input features x base on a pre-assumed probability distribution function of the input. To generate new samples, we just need to sample from the probability distribution function of the input.

- `Implicit model`: a simulator model that can generate new samples that look like the training data. New samples are generated directly from the model without any pre-assumed probability distribution function of the input.

<p align="center">
  <img src="https://imgur.com/MelJzGj.png" >
  <br>
  <i>Types of Generative models</i>
</p>

GAN is an implicit model because it is a simulator model that can generate new samples that look like the training data. New samples are generated directly from the model without any pre-assumed probability distribution function of the input.

## **2. GANs architecture**

### **2.1. GANs intuition**

GANs intuition based on `zero-sum non-cooperative game`.

- Non-cooperative game - Players act independently and selfishly without cooperating with each other.
- Zero-sum - The total benefit of all players in the game always equals zero or the reward of one player is exactly balanced by the losses of the other(s). So one player can only benefit at the expense of others.
- Players' goals are opposed - the benefits of one player must harm the other(s). There is no possibility of a win-win outcome for all players.
- At a time, the game will reach a equilibrium point where neither of them can improve their benefit (or loss) by changing their strategy. We call that point `Nash equilibrium`.

In GANs, we have 2 players: Generator and Discriminator.

- The generator network takes random input (noise) and generates samples, such as images, text, or audio that look same as the training data it was trained on. The goal of the generator is to produce samples that are indistinguishable from real data.

- The discriminator network tries to distinguish between real and generated samples. It is trained with real samples from the training data and generated samples from the generator. Its objective is to correctly classify real data as real and generated data as fake.

The training process is an adversarial game between the generator and the discriminator.

- The generator aims to produce more realistic samples that fool the discriminator.
- The discriminator tries to improve its ability to distinguish between real and generated data.
- This adversarial training will improve both over time.
- Ideally, this process converges to a point where the generator is capable of generating high-quality samples that are difficult for the discriminator to distinguish from real data (Nash equilibrium).

<p align="center">
  <img src="https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/11/d_rk.png?resize=768%2C357&ssl=1" >
  <br>
  <i>GANs flowchart</i>
</p>

<p align="center">
  <img src="https://dz2cdn1.dzone.com/storage/temp/10276123-dzone.png" >
  <br>
  <i>Fake money example</i>
</p>

### **2.2. Generator**

<p align="center">
  <img src="https://imgur.com/a4p9G3d.png" >
  <br>
  <i>Generator architecture example</i>
</p>

In basic, Generator is a neural network that takes random input (noise) and generates samples, such as images, text, or audio that look same as the training data it was trained on.

Input noise is intialized ramdomly from a normal distribution (Gaussian distribution) or uniform distribution. In some modern GANs, input can be images, text, or audio. But in original GANs paper, input is random noise.

From the input ramdom noise z, generater is a deep neural network that generates new samples that look like the training data. This is done by transforming the input noise z into a sample that has the same shape as the training data. Then these fake samples are fed into the Discriminator.

### **2.3. Discriminator**

<p align="center">
  <img src="https://imgur.com/vGjX6DM.png" >
  <br>
  <i>Discriminator architecture example</i>
</p>

Discriminator is a neural network that tries to distinguish between real and generated samples. It is trained with real samples from the training data and generated samples from the generator. Its objective is to correctly classify real data as real and generated data as fake.

Label is real if input data is real data from training data. Label is fake if input data is generated data from generator. So discriminator is simply a binary classifier.

<p align="center">
  <img src="https://cdn.clickworker.com/wp-content/uploads/2022/11/Generative-Adversarial-Networks-Architecture-scaled.jpg" >
  <br>
  <i>Example of GANs architecture</i>
</p>

### **2.4. Loss function**

#### **2.4.1. Review cross entropy loss**

For binary classification problems, we have the loss function for a `single training` instance:

```math
c(\theta) = \begin{cases}
-log(\hat{p}) & \text{if } y = 1, \\
-log(1 - \hat{p}) & \text{if } y = 0.
\end{cases}
```

This loss function makes sense because:

- $`-log(t)`$ grows very large when $`t`$ approaches 0 and approaches 0 when $`t`$ approaches 1.
- The cost will be large if the model make a wrong prediction
  - $`y = 0`$ but model estimates $`\hat{p}`$ close to 1
  - $`y = 1`$ but model estimates $`\hat{p}`$ close to 0
- The cost will be close to 0 if the model makes a right prediction

The cost function over `the whole training set` is simply the average cost over all training instances (called the **log loss**):

```math
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y_i log(\hat{p}_i) + (1 - y_i) log(1 - \hat{p}_i)]
```

This cost function is convex, so Gradient Descent (or any other optimization algorithm) is guaranteed to find the global minimum (if the learning rate is not too large and you wait long enough).

<p align="center">
  <img src="https://raw.githubusercontent.com/shruti-jadon/Data_Science_Images/main/cross_entropy.png" >
  <br>
  <i>Log Function</i>
</p>

#### **2.4.2. Discriminator loss**

The discriminator is a binary classifier to distinguish if the input $`x`$ is real (from real data) or fake (from the generator). Typically, the discriminator outputs a scalar prediction $`o\in\mathbb R`$ for each input $`x`$, by apply sigmoid function to obtain the predicted probability:

```math
D(\mathbf x) = o = \frac{1}{1 + e^{-\mathbf{w}^\intercal\mathbf{x}}}
```

Assume the label y for the true data is 1 and 0 for the fake data (generated data) and we asume that generator G is fixed when training Discriminator. Our objective is to minimize the cross-entropy loss:

```math
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y_i log(\hat{p}_i) + (1 - y_i) log(1 - \hat{p}_i)]
```

In which:

- $`\hat{p}_i = D(\mathbf x_i)`$ is the predicted probability of the discriminator for the input $`\mathbf x_i`$.
- $`y_i`$ is the label of the input $`\mathbf x_i`$. If $`\mathbf x_i`$ is a real data, then $`y_i = 1`$ and $`\mathbf x_i`$ is a fake data, then $`y_i = 0`$.

So, we have 2 cases for each input $`\mathbf x_i`$:

- If $`\mathbf x_i`$ is a real data, then $`y_i = 1`$ and $`\hat{p}_i = D(\mathbf x_i)`$. So the loss function is:

```math
J(\theta) = - [1  \times  log(D(\mathbf x_i)) + (1 - 1)  \times  log(1 - D(\mathbf x_i))] = -log(D(\mathbf x_i))
```
  
- If $`\mathbf x_i`$ is a fake data, then $`y_i = 0`$ and we have we have $`\mathbf x_i = G(z_i)`$ so $`\hat{p}_i = D(G(z_i))`$. So the loss function is:

```math
J(\theta) = - [0 \times log(D(\mathbf x_i)) + (1 - 0) \times log(1 - D(\mathbf x_i))] = -log(1 - D(G(z_i)))
```

The objective of Discriminator is to maximize $`D(\mathbf x)`$ and minimize $`D(G(\mathbf z))`$ by minimizing the following objective function:

```math
\min_{D} V(D) = - \underbrace{\mathbb{E}_{x \sim p_{data}(x)} [\log D(x)]}_{\text{log-probability that D predict x is real}} - \underbrace{\mathbb{E}_{z \sim p_{z}(z)} [\log (1-D(G(z)))]}_{\text{log-probability D predicts G(z) is fake}}
```

Inverting the sign, we can maximize the following objective function:

```math
\max_{D} V(D) = \underbrace{\mathbb{E}_{x \sim p_{data}(x)} [\log D(x)]}_{\text{log-probability that D predict x is real}} + \underbrace{\mathbb{E}_{z \sim p_{z}(z)} [\log (1-D(G(z)))]}_{\text{log-probability D predicts G(z) is fake}}
```

In the above equation:

- $`E_{x \sim p_{data}(x)}`$ is the expectation of $`x`$ drawn from the real data distribution $`p_{data}(x)`$. In other words, it is the average of $`\log D(x)`$ over all real data $`x`$.
- $`E_{z \sim p_{z}(z)}`$ is the expectation of $`z`$ drawn from the noise distribution $`p_{z}(z)`$. In other words, it is the average of $`\log (1-D(G(z)))`$ over all fake data $`G(z)`$. Note that, $`G(z)`$ is the output of the generator.

#### **2.4.3. Generator loss**

The goal of training Generator is to strengthen the Generator's image creation ability so that the image it produces is as realistic as possible. We asume in this phase, the discriminator is fixed.

For the generator, it first draws some parameter $`\mathbf z\in\mathbb R^d`$ from a source of randomness, e.g., a normal distribution $`\mathbf z \sim \mathcal{N} (0, 1)`$. We often call z as the `latent variable`.

It then applies a function to generate $`\mathbf x'=G(\mathbf z)`$. The goal of the generator is to fool the discriminator to classify $`\mathbf x'=G(\mathbf z)`$ as true data, i.e., we want $`D( G(\mathbf z)) \approx 1`$.

In other words, for a given fixed discriminator D, we update the parameters of the generator G to maximize the cross-entropy loss when y = 0. Which means we try to maximize $`D(G(\mathbf z))`$:

```math
\max_G \{ - (1-y) \log(1-D(G(\mathbf z))) \} = \max_G \{ - \log(1-D(G(\mathbf z))) \}.
```

If the generator does a perfect job, then $`D(\mathbf x')\approx 1`$  so the above loss is near 0, which results in the gradients that are too small to make good progress for the discriminator.

Invert the sign of the above equation, we can minimize the following objective function:

```math
\min_G V(G) = \underbrace{\mathbb{E}_{z \sim p_{z}(z)} [\log (1-D(G(z)))]}_{\text{log-probability D predicts G(z) is fake}}
```

#### **2.4.4. GANs loss function (min-max GANs loss)**

To sum up, D and G are playing a “minimax” game with the comprehensive objective function:

```math
\min_{G} \max_{D} V(D, G) = \underbrace{\mathbb{E}_{x \sim p_{data}(x)} [\log D(x)]}_{\text{log-probability that D predict x is real}} + \underbrace{\mathbb{E}_{z \sim p_{z}(z)} [\log (1-D(G(z)))]}_{\text{log-probability D predicts G(z) is fake}}
```

In which:

- D try to maximize the objective function by maximizing $`V(D, G)`$. Which mean it updates parameters to maximize value of $`D(x)`$ and minimize value of $`D(G(z))`$. In other words, it tries to classify real data as real and fake data as fake.
- G try to minimize the objective function by minimizing $`V(D, G)`$. Which mean it updates parameters to maximize value of $`D(G(z))`$. In other words, it tries to generate fake data that look like real data.

### **2.5. Training process**

In training process, we alternate between training the discriminator and the generator. In each step, we update the discriminator by one step and then update the generator by one step.

<p align="center">
  <img src="https://machinelearningmastery.com/wp-content/uploads/2019/05/Summary-of-the-Generative-Adversarial-Network-Training-Algorithm-1024x669.png" >
  <br>
  <i>GANs adversarial training process</i>
</p>

Note that when train the discriminator, we need to freeze the parameters of generator and vice versa. The training process is continued until the discriminator cannot distinguish between real and fake data or we reach the maximum number of epochs.

#### **Generator diminished gradient**

In training process, we update the discriminator by one step and then update the generator by one step. However, the generator may not be able to learn well if the discriminator is too good. This is because the gradients of the generator will be very small when the discriminator is too good. That means:

```math
\frac{1}{m}\nabla_{\theta_{G}} \sum_{i=1}^{m} \log (1-D(G(z^{(i)}))) \approx 0
```

To solve that problems, we change the objective function that will generator maximizes the log of the discriminator probabilities

```math
\max_G V(G) = \underbrace{\mathbb{E}_{z \sim p_{z}(z)} [\log (D(G(z)))]}_{\text{log-probability D predicts G(z) is real}}
```

<p align="center">
  <img src="https://images.viblo.asia/f42d99cb-3f73-428a-ac3b-1df7fbdba49d.jpeg" >
  <br>
  <i>Generator change</i>
</p>

This change is inspired by framing the problem from a different perspective, where the generator seeks to maximize the probability of images being real, instead of minimizing the probability of an image being fake.

So, the backpropagation process will be:

<p align="center">
  <img src="https://miro.medium.com/max/1600/1*M_YipQF_oC6owsU1VVrfhg.jpeg" >
  <br>
  <i>Backpropagation process</i>
</p>

## **3. Applications and challenges of GANs**

Application of GANs and GANs variants model:

- Image generation
- Image to image translation
- Text to image generation
- Image super-resolution
- ...

Chanllenges of GANs:

- GANs tend to show some inconsistencies in performance.
- Mode collapse
- Vanishing gradient
- Convergence
- Can't control the output (conditioning, style,...) which will be solved by GANs variants model.
