# **DEEP LEARNING PART 1**

## **1. Linear Neural Network**

### **1.1. Linear Neural Network for Regression**

#### **a. Review Linear Regression**

- Linear regression model makes a prediction by simply computing a weighted sum of the input features, plus a constant called the bias term (also called the intercept term).

- Linear regression model prediction:

```math
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
\newline
or
\newline
\hat{y} = h_\theta(x) = \theta^T \cdot x
```

- Cost function of Linear Regression:

```math
MSE(X, h_\theta) = \frac{1}{m} \sum_{i=1}^{m} (\theta^T \cdot x^{(i)} - y^{(i)})^2
```

- The Normal Equation (the closed-form solution of Linear Regression)

```math
\hat{\theta} = (X^T \cdot X)^{-1} \cdot X^T \cdot y
```

- Stochastic Gradient Descent update rule for Linear Regression:

```math
\theta_{j+1} = \theta_j - \eta (\hat{y}_i - y_i)x^{(i)}_j
```

- In the above equations:
  - $`\hat{y}`$ is the predicted value.
  - $`n`$ is the number of features.
  - $`x_i`$ is the $`i^{th}`$ feature value.
  - $`\theta_j`$ is the $`j^{th}`$ model parameter (including the bias term $`\theta_0`$ and the feature weights $`\theta_1, \theta_2, \dots, \theta_n`$).

#### **b. Linear Regression as a Neural Network**

- Linear regression can be considered as a specific case of a neural network, in which every feature is represented by an input neuron, all of which are connected directly to the output. Structure and Layers of Linear Regression as a Linear Neural Network:
  - Input Layer: represents the features as independent variables of the data. The number of nodes in the input layer is equal to the number of features in the data.
  - Output Layer: consists of a single neuron, representing the predicted output (also known as the dependent variable or target variable).

![Linear Regression as a Linear Neural Network](https://joshuagoings.com/assets/linear.png)

### **1.2. Linear Neural Network for Classification**

#### **1.2.1. Logistic Regression**

##### **a. Review Logistic Regression**

- `Logistic Regression` is a classification algorithm use `Sigmod function` to predict the output of discrete values (0/1, yes/no, true/false) based on given set of independent variable(s).

- Sigmoid function:

```math
\sigma(t) = \frac{1}{1 + e^{-t}}
```

- Estimating Probabilities

  ```math
  \hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}
  ```

  - In which, z is the linear function of the input features:

  ```math
  z = \theta^T \cdot x = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
  ```

  - Make prediction:

  ```math
  \hat{y} = \begin{cases}
  0 & \text{if } \hat{p} < 0.5, \\
  1 & \text{if } \hat{p} \geq 0.5.
  \end{cases}
  ```

- Training and optimizing cost function:

  - Cost function for a `single training` instance:

  ```math
  c(\theta) = \begin{cases}
  -log(\hat{p}) & \text{if } y = 1, \\
  -log(1 - \hat{p}) & \text{if } y = 0.
  \end{cases}
  ```

  - The cost function over `the whole training set`:

  ```math
  J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y_i log(\hat{p}_i) + (1 - y_i) log(1 - \hat{p}_i)]
  ```

  - Update equation for Stochastic Gradient Descent:

  ```math
  \mathbf{\theta}^* = \mathbf{\theta} - \eta(\hat{p}_i - y_i)\mathbf{x}_i
  ```

##### **b. Logistic Regression as a Neural Network**

- Logistic Regression can ve viewed as a shallow Neural Network. `Sigmoid function` is used to transform the input features, it is also called the `activation function` of the neuron. Logistic Regression have only one output neuron, which is used to predict the probability of the input belonging to a particular class.

![Logistic Regression Neural Network](https://deeplearningmath.org/images/shallow_NN.png)

#### **1.2.2. Softmax Regression**

##### **a. Review Softmax Regression**

- `Softmax Regression` is a generalization of logistic regression to the case where we want to handle multiple classes.

- Main idea:
  - When given an instance `x`, the model first computes a score $`s_k(x)`$ for each class `k`.
  
  ```math
  s_k(x) = \theta_k^T \cdot x
  ```

  - After computed the score of every class for the instance `x`, estimate the probability $`\hat{p}_k`$ that the instance belongs to class `k` by running the scores through the `softmax function`.

  ```math
  \hat{p}_k = \sigma(s(x))_k = \frac{exp(s_k(x))}{\sum_{j=1}^{K} exp(s_j(x))}
  ```

  - In this equation:
    - `K` is the number of classes.
    - $`s(x)`$ is a vector containing the scores of each class for the instance `x`.
    - $`\sigma(s(x))_k`$ is the estimated probability that the instance `x` belongs to class `k` given the scores of each class for that instance.
    - If we have two classes (K = 2), then Softmax Regression is equivalent to Logistic Regression.

- Like the Logistic Regression, Softmax Regression also predicts the class with the highest estimated probability:

```math
\hat{y} = \underset{k}{\text{argmax }} \sigma(s(x))_k = \underset{k}{\text{argmax }} s_k(x) = \underset{k}{\text{argmax }} (\theta_k^T \cdot x)
```

- Cost Function - Cross Entropy

  - The cross entropy cost function is given by:

  ```math
  J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} log(\hat{p}_k^{(i)})
  ```

  - $`y_k^{(i)}`$ is the target probability that the $`i^{th}`$ instance belongs to class `k`. In general, it is equal to 1 if the target class for the $`i^{th}`$ instance is `k`; otherwise, it is equal to 0.
  - m is the number of instances in the dataset.

- Cross entropy gradient vector for class `k`:

  ```math
  \nabla_{\theta_k} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (\hat{p}_k^{(i)} - y_k^{(i)}) \mathbf{x}^{(i)}
  ```

- If we use Stochastic Gradient Descent, with a pair of training instances $`(x_i, y_i)`$, the update equation for matrix $`\theta`$ is:

  ```math
  \mathbf{\theta} = \mathbf{\theta} - \eta \mathbf{x}_{i}(\mathbf{\hat{p}}_i - \mathbf{y}_i)^T
  ```

##### **b. Softmax Regression as a Neural Network**

- `one-vs-rest`:
  - A binary classification model can be used to estimate the probability that an instance belongs to the positive class.
  - If there are `C` classes, then we need to train `C` binary classifiers, one for each class.
  - When classify an instance, we get the decision score from all classifier for that instance and select the highest score.

![One vs rest](https://machinelearningcoban.com/assets/13_softmax/onevsrest.png)

- The main problem of `one-vs-rest`:
  - Not able to output the probability that an instance belongs to each class.
  - No connections between the output of the classifiers, its sum can be smaller or larger than 1.

- To fix this problem, we can use the softmax regression model. After having $`z_1, z_2, ..., z_k`$ we can compute the probability that the instance belongs to class `k` by running the scores through the softmax function:

```math
a_i = \frac{\exp(z_i)}{\sum_{j=1}^C \exp(z_j)}, ~~ \forall i = 1, 2, \dots, C
```

![Softmax Regression](https://machinelearningcoban.com/assets/13_softmax/softmax_nn.png)

## **2. Neural Network**

### **2.1. Overview**

- **Neural Network**, also called **Artificial Neural Network** (ANN) is a type of AI and Machine Learning model that is inspired by human brain. It consists of a large number of nodes (neurons) that are interconnected and exchange information.

- A simple architecture of a neural network is shown below:

![Neural Network](https://www.researchgate.net/publication/320069212/figure/fig5/AS:543453192912901@1506580746824/Simple-neural-network-architecture.png)

- In the above figure:

  - **Nodes (or Neurons)** are the basic units of a neural network. Each node receives inputs from the previous layer, performs some computation, and passes the output to the next layer. Each node has a weight and a bias. The weight represents the strength of the connection between the nodes and based on the weights, the neurons perform computations. The bias is a constant value that is added to the input of the node. The bias allows you to shift the activation function curve up or down. The bias is also a trainable parameter and is updated during the training process.

  - **Weights** are the connections between the nodes. The weight represents the strength of the connection between the nodes and based on the weights, the neurons perform computations.

  - **Layers** are the collection of nodes. There are 3 main types of layers:

    - **Input layer**: receives the raw data and inputs it into the network. Consists of nodes that represent the features of the input data.

    - **Hidden layers** is one or more layers between the input and output layers. The hidden layers perform computations and pass the output to the next layer. The hidden layers are responsible for extracting the features from the input data.

    - **Output layer** is the final layer of the network. It receives the input from the previous hidden layer, performs computations, and generates the output.

- Each model always has an input layer, an output layer, with or without hidden layers. The total number of layers in the model is conventionally the number of layers – 1 (Not counting input layers).

### **2.2. Computing in Neural Network**

- Consider the following neural network:

![Neural Network](https://images.viblo.asia/f278a993-05a6-4725-a77f-f0d802e56cd1.png)

- Given an input, neuron provides the output and passes that output as an input to the subsequent layer. A neuron can be thought of as a combination of 2 parts:
  - The first part performs a linear transformation using input and weights.
  - The second part performs an activation function on the output of the first part. The output of the second part is the output of the neuron.

![Neuron](https://images.viblo.asia/baa46a6f-1096-4754-a226-6c1fc19edc68.png)

- The hidden layer comprises of various neurons, each of which performs the above 2 calculations. The 4 neurons present in the hidden layer of our shallow neural network compute the following:

![Hidden layer](https://images.viblo.asia/3551d7c8-df22-42d6-bdcd-66554dd8ca70.png)

- In the above equation:
  - The superscript number `[i]` denotes the layer number and the subscript number `j` denotes the neuron number in a particular layer.
  - $`a^{[i]}_j`$ is the output of the $`j^{th}`$ neuron in the $`i^{th}`$ layer.
  - $`w^{[i]}_j`$ is the weight vector of the $`j^{th}`$ neuron in the $`i^{th}`$ layer.
  - $`b^{[i]}_j`$ is the bias of the $`j^{th}`$ neuron in the $`i^{th}`$ layer.
  - $`z^{[i]}_j`$ is the linear transformation of the $`j^{th}`$ neuron in the $`i^{th}`$ layer.

- So with input X, the output of layer 1 and 2 is:

![Output of layer 1 and 2](https://images.viblo.asia/9742b40c-2164-4498-bf55-49fccbc2c705.PNG)

- We can vectorize the above equation to decrease the computation time:

![Vectorize](https://images.viblo.asia/acd77a72-949c-40a2-a2bf-4558e612939d.PNG)

- The above process is called `forward propagation`. The forward propagation is used to compute the output of the neural network for a given input. The output of the neural network is compared with the actual output and the error is calculated. The error is then used to update the weights and biases of the neural network. This process is called `backpropagation`.

### **2.3. Activation Function**

- Activation functions are mathematical equations that determine the output of a neural network. Some activation functions are:

  - **Sigmoid function**:

  ```math
  \sigma(z) = \frac{1}{1 + e^{-z}}
  ```

  - **Tanh function**:

  ```math
  tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
  ```

  - **ReLU function**:

  ```math
  ReLU(z) = max(0, z)
  ```

  - **Leaky ReLU function**:

  ```math
  LeakyReLU(z) = max(\alpha z, z)
  ```

![Activation functions](https://ai-artificial-intelligence.webyes.com.br/wp-content/uploads/2022/09/image-1-967x1024.png)

- The above activation function are non-linear functions. Non-linear actovation function help:
  - Learn complex patterns in the data.
  - Represent complex decision boundaries.
  - ...

- Why we don't use linear activation function? -> If we use linear activation function, the output of the neural network will be a linear function of the input. So, no matter how many layers we add, the output will be a linear function of the input. So, we will not be able to learn complex patterns in the data.

### **2.4. Gradient Descent for Neural Network**

#### **Backpropagation**

- `Backpropagation` is used to calculate the gradient of the loss function with respect to the weights of the network. This gradient is then used to update the weights of the network using an optimization algorithm such as gradient descent.

- `Backpropagation` works by recursively applying the chain rule of calculus to calculate the gradients of the loss function with respect to the weights of the network. The process starts at the output layer of the network and works backwards towards the input layer.

- Consider a log loss:

![Loss function](https://miro.medium.com/v2/resize:fit:640/format:webp/1*yXVVHG47pTXfNyJ60dEd6Q.png)

- Applying the chain rule of calculus, we can calculate the gradient of the loss function with respect to the weights of the network:

![Gradient of loss function](https://miro.medium.com/v2/resize:fit:720/format:webp/1*EPTVQQkVZGx3ABc8D0y-pQ.png)

- In short:
  - Feedforward: calculate the output of the neural network for a given input. Then calculate the error.

  ![Feedforward](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/fw.png?w=1065&ssl=1)

  - Backpropagation: calculate the gradient of the loss function with respect to the weights of the network.

  ![Backpropagation](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/backpropagation-2.png?w=1127&ssl=1)

![Forward and backward](https://www.baeldung.com/wp-content/uploads/sites/4/2022/05/epoch-fwd-bwd-pass.png)

#### **Gradient Descent**

- After calculating the gradient of the loss function with respect to the weights of the network, we can update the weights using the gradient descent algorithm. The steps of gradient descent can be summarized as:

![Gradient Descent](https://images.viblo.asia/dafb2448-7671-4896-aecb-b38b1c9a0d4f.PNG)

### **2.5 Weight Initialization**

- Weight initialization is the process of setting the initial values of the weights of the neural network. The weights of the neural network must be initialized to small random numbers. If the weights are initialized to large random numbers, the output of the activation function will be large and the gradients will be close to zero. This is called the `vanishing gradient problem`.

- If we initialize the weights to zero, there are two problems:
  - All the neurons will have the same weights and the same gradients. So, all the neurons will be symmetric and they will always learn the same thing. This is called the `symmetry breaking problem`.
  - It's also can cause `dead neurons` problem. If the input of a neuron is zero, then the gradient of the neuron will be zero. So, the weights of the neuron will not be updated and the neuron will not learn anything.

- The best method of initialization is Xavier’s Initialization. Mathematically it is defined as:

![Xavier’s Initialization](https://miro.medium.com/v2/resize:fit:640/format:webp/1*NTL6KFMmRa_aW2k6WQGNkw.png)

- It states that Weight Matrix W of a particular layer l are picked randomly from a normal distribution with mean $`\mu = 0`$ and variance $`\sigma^2 = \frac{1}{n_{l-1}}`$ (multiplicative inverse of the number of neurons in layer l−1). The bias b of all layers is initialized with 0.
