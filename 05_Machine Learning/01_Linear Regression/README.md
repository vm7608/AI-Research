# **MACHINE LEARNING PART 2**

## **1. Linear Regression**

### **1.1. Introduction**

- Linear regression model makes a prediction by simply computing a weighted sum of the input features, plus a constant called the bias term (also called the intercept term).

![Linear Regression](https://static.javatpoint.com/tutorial/machine-learning/images/linear-regression-in-machine-learning.png)

- Linear regression model prediction:

```math
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n
```

- In this equation:

  - $`\hat{y}`$ is the predicted value.
  - *n* is the number of features.
  - $`x_i`$ is the $`i^{th}`$ feature value.
  - $`\theta_j`$ is the $`j^{th}`$ is model parameter, included:
    - $`\theta_0`$ is the bias.
    - $`\theta_1, \theta_2, ..., \theta_n`$ is the feature weights.

- Explain:
  - We need the bias term $`\theta_0`$ because of the following reasons:
    - If $`x_1, x_2, ..., x_n`$ are all equal to 0, then the predicted value is equal to $`\theta_0`$.
    - Bias allows to shift the line of linear regression up or down to better fit the data. If $`\theta_0`$ is positive, the line will shift up, and vice versa. Without bias, the line will always pass through the origin (0, 0) and we only able to control the slope of the line.

- This can be written much more concisely using a vectorized form:

```math
\hat{y} = h_\theta(x) = \theta^T \cdot x
```

- In this equation:

  - $`\theta`$ is the model's *parameter vector*, containing the bias term $`\theta_0`$ and the feature weights $`\theta_1`$ to $`\theta_n`$.
  - $`\theta^T`$ is the transpose of $`\theta`$ (a row vector instead of a column vector).
  - x is the *feature vector*, containing $`x_0, x_1, ..., x_n`$, with $`x_0`$ always equal to 1 (because $`\theta_0`$ is the bias term, not a feature)
  - $`\theta^T . x`$ is the dot product of $`\theta^T`$ and `x`.
  - $`h_\theta`$ is the hypothesis function (Predictive function), using the model parameters $`\theta`$.

- To train a Linear Regression model, we need to find the value of $`\theta`$ that minimizes the RMSE. In practice, it is simpler to minimize the MSE because it leads to the same result (the value of $`\theta`$ that minimizes the MSE also minimizes the RMSE).

- The MSE of a Linear Regression $`h_\theta`$ on a training set `X`:

```math
MSE(\theta) = MSE(X, h_\theta) = \frac{1}{m} \sum_{i=1}^{m} (\theta^T \cdot x_i - y_i)^2 = \frac{1}{m} \sum_{i=1}^{m} (\hat{y_i} - y_i)^2
```

- The above equation is the **cost function** of a Linear Regression model. And the goal is to minimize this cost function. MSE is the most commonly used as cost function of linear regression model because:
  - It's differentiable (khả vi), so we can use gradient descent to find the minimum.
  - It penalizes large errors more than smaller errors (due to the square operation).
  - For a linear regression model, the MSE cost function has a *bowl shape* (convex function), so there is no local minima. There is just one global minimum.

### **1.2. The Normal Equation**

- To find the value of $`\theta`$ that minimizes the cost function, there is a closed-form solution - in other words, a mathematical equation that gives the result directly. This is called the **Normal Equation**:

```math
\hat{\theta} = (X^T \cdot X)^{-1} \cdot X^T \cdot y
```

- How we get the Normal equation?
  - First, we have the vectorized form of cost function:

  ```math
  MSE(\theta) = \frac{1}{m} \sum_{i=1}^{m} (\theta^T \cdot x_i - y_i)^2
  ```

  - After that, we can find the value of $`\theta`$ that minimizes the cost function by taking the derivative of the cost function with respect to $`\theta`$ and set it to 0:

  ```math
  \frac{\partial}{\partial \theta} MSE(\theta) = \frac{2}{m} X^T \cdot (X \cdot \theta - y) = 0
  ```

  - After that, we can solve for $`\theta`$:

  ```math
  X^T \cdot (X \cdot \theta - y) = 0
  ```

  ```math
  X^T \cdot X \cdot \theta = X^T \cdot y
  ```

  - Finally, we can get the Normal Equation:

  ```math
  \hat{\theta} = (X^T \cdot X)^{-1} \cdot X^T \cdot y
  ```

- In this equation:

  - $`\hat{\theta}`$ is the value of $`\theta`$ that minimizes the cost function.
  - y is the vector of target values containing $`y_1,...,y_m`$
  - $`X`$ is the matrix of feature values containing $`x_0, x_1, ..., x_m`$. Each row is a vector of feature values for one instance, with $`x_0`$ always equal to 1.

- Instead of using Gradient Descent, we can use the Normal Equation to solve for the optimal regression coefficients (weights) in one step.

- Advantage of Normal Equation:
  - It solves for the exact optimal coefficients in one step. No need to iterate like Gradient Descent.
  - It is more efficient for smaller datasets. For large datasets, inverting the $`X^T \cdot X`$ matrix can be computationally expensive.

- Disadvantage of Normal Equation:
  - It is computationally expensive when the number of features grows large (e.g. 100,000).
  - Can cause numerical problems when computing the inverse of $`X^T \cdot X`$ (i.e. when the matrix is not invertible).
