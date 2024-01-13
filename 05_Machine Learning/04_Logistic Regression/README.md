# **MACHINE LEARNING PART 2**

## **4. Logistic Regression**

### **4.1. Introduction**

- `Logistic Regression` is a classification algorithm use `Sigmod function` to predict the output of discrete values (Binary values like 0/1, yes/no, true/false) based on given set of independent variable(s).

### **4.2. Sigmod Function**

- The sigmoid function has the following formula:

```math
\sigma(t) = \frac{1}{1 + e^{-t}}
```

![sigmoid](https://lh6.googleusercontent.com/9IScScwpevNcZdwXsvwV7yR8ighTYvSzjPI85zZEiblYw8PFYWcO6BUFZfgDefPcWiUXkliPv7tUXdW5gQiR2aRVUwPveFIni7Qt76ZkqdLTXViU-iGQdYtg2FfUEyqUOO-ujtNP)

- Some properties of the sigmoid function:
  - Its outputs are in range (0, 1)
  - It is a differentiable function (khả vi), so we can use gradient descent to find the minimum of the cost function.
  - Gradient of the sigmoid function is easy to calculate:

  ```math
  \sigma'(t) = \sigma(t)(1 - \sigma(t))
  ```

### **4.3. Estimating Probabilities**

- Logistic Regression model estimated probability:

```math
\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}
```

- In which, z is the linear function of the input features:

```math
z = \theta^T \cdot x = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
```

- That why logistic regression is considered a linear model because the log-odds of the outcome variable is a linear combination of the input variables.

- Once the Logistic Regression model has estimated the probability $`\hat{p} = h_{\theta}(x)`$ that an instance x belongs to the positive class, it can make its prediction $`\hat{y}`$ easily:

```math
\hat{y} = \begin{cases}
0 & \text{if } \hat{p} < 0.5, \\
1 & \text{if } \hat{p} \geq 0.5.
\end{cases}
```

- Notice that $`\sigma(t) < 0.5`$ when $`t < 0`$, and $`\sigma(t) \geq 0.5`$ when $`t \geq 0`$, so a Logistic Regression model predicts 1 if $`\theta^T \cdot x`$ is positive, and 0 if it is negative.

### **4.4. Training and Cost Function**

- So, we have the cost function for a `single training` instance:

```math
c(\theta) = \begin{cases}
-log(\hat{p}) & \text{if } y = 1, \\
-log(1 - \hat{p}) & \text{if } y = 0.
\end{cases}
```

- This loss function makes sense because:
  - $`-log(t)`$ grows very large when $`t`$ approaches 0 and approaches 0 when $`t`$ approaches 1.
  - The cost will be large if the model make a wrong prediction
    - $`y = 0`$ but model estimates $`\hat{p}`$ close to 1
    - $`y = 1`$ but model estimates $`\hat{p}`$ close to 0
  - The cost will be close to 0 if the model makes a right prediction

![Log Function](https://www.rapidtables.com/math/algebra/ln/ln-graph.png)

- The cost function over `the whole training set` is simply the average cost over all training instances (called the **log loss**):

```math
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y_i log(\hat{p}_i) + (1 - y_i) log(1 - \hat{p}_i)]
```

- This cost function is convex, so Gradient Descent (or any other optimization algorithm) is guaranteed to find the global minimum (if the learning rate is not too large and you wait long enough).

#### **Optimizing cost using Stochastic Gradient Descent:**

- We have the cost function of a single training instance $`(x_i, y_i)`$:

```math
J(\mathbf{\theta}; \mathbf{x}_i, y_i) = -[y_i log(\hat{p}_i) + (1 - y_i) log(1 - \hat{p}_i)]
```

- The gradient of above cost function is:

```math
\frac{\partial J(\mathbf{\theta}; \mathbf{x}_i, y_i)}{\partial \mathbf{\theta}} = -(\frac{y_i}{\hat{p}_i} - \frac{1- y_i}{1- \hat{p}_i} ) \frac{\partial \hat{p}_i}{\partial \mathbf{\theta}} = \frac{\hat{p}_i - y_i}{\hat{p}_i(1 - \hat{p}_i)} \frac{\partial \hat{p}_i}{\partial \mathbf{\theta}}
```

- We have $`z = \mathbf{\theta}^T\mathbf{x}`$ and $`\hat{p} = \sigma(z)`$, so the gradient of sigmoid function:

```math
\frac{\partial \hat{p}_i}{\partial \mathbf{\theta}} = \frac{\partial \hat{p}_i}{\partial z} \frac{\partial z}{\partial \mathbf{\theta}} = \frac{\partial \hat{p}_i}{\partial z} \mathbf{x} =\hat{p}_i(1 - \hat{p}_i) \mathbf{x}
```

- From the graph of sigmoid function, we have the final gradient:

```math
\frac{\partial J(\mathbf{\theta}; \mathbf{x}_i, y_i)}{\partial \mathbf{\theta}} = (\hat{p}_i - y_i)\mathbf{x}_i
```

- So we have the update rule for each iteration of SGD:

```math
\mathbf{\theta} = \mathbf{\theta} - \eta(\hat{p}_i - y_i)\mathbf{x}_i
```

### **4.5. Decision Boundaries**

- We have to choose a threshold value to make prediction. If $`\hat{p} \geq 0.5`$, then the model predicts 1, or else it predicts 0. So the decision boundary is where $`\hat{p} = 0.5`$, which is equivalent to $`\theta^T \cdot x = 0`$.

- Just like the other linear models, Logistic Regression models can be regularized using $`\ell_1`$ or $`\ell_2`$ penalties.

- The hyperparameter controlling the regularization strength of a Scikit-Learn LogisticRegression model is inverse of alpha: $`C`$. The higher the value of $`C`$, the less the model is regularized.

### **4.6. Neural Networks approach**

- Logistic Regression can be implemented using a very simple neural network consisting of a single neuron layer.

![Logistic Regression Neural Network](https://deeplearningmath.org/images/shallow_NN.png)

### **4.7. Softmax Regression**

- `Softmax Regression` is a generalization of logistic regression to the case where we want to handle multiple classes. Logistic Regression is a specific case of Softmax Regression when number of classes is 2.

- Main idea:
  - When given an instance `x`, the Softmax Regression model first computes a score $`s_k(x)`$ for each class `k`. The equation to compute $`s_k(x)`$ look familiar with the equation for Linear Regression prediction:
  
  ```math
  s_k(x) = \theta_k^T \cdot x
  ```
  
  - Note that each class has its own dedicated parameter vector $`\theta_k`$.

  - After computed the score of every class for the instance `x`, you can estimate the probability $`\hat{p}_k`$ that the instance belongs to class `k` by running the scores through the `softmax function`. The scores are generally called logits or log-odds:

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

#### **Cost Function - Cross Entropy**

- The cost function of Softmax Regression called the **cross entropy**. Cross entropy is frequently used to measure how well a set of estimated class probabilities match the target classes.

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

#### **Softmax Regression Neural Networks approach**

- Review the `one-vs-rest` model:
  - A binary classification model can be used to estimate the probability that an instance belongs to the positive class.
  - If there are `N` classes, then we need to train `N` binary classifiers, one for each class.
  - When classify an instance, we get the decision score from all classifier for that instance and select the highest score.

![One vs rest](https://machinelearningcoban.com/assets/13_softmax/onevsrest.png)

- The main problem of above model:
  - Not able to output the probability that an instance belongs to each class.
  - No connections between the output of the classifiers, its sum can be smaller or larger than 1, and the highest score can be negative.

- To fix this problem, we can use the softmax regression model. After having $`z_1, z_2, ..., z_k`$ we can compute the probability that the instance belongs to class `k` by running the scores through the softmax function:

```math
a_i = \frac{\exp(z_i)}{\sum_{j=1}^C \exp(z_j)}, ~~ \forall i = 1, 2, \dots, C
```

![Softmax Regression](https://machinelearningcoban.com/assets/13_softmax/softmax_nn.png)
