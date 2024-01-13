# **MACHINE LEARNING PART 2**

## **3. Regularized Linear Models**

### **3.1. Introduction**

- **Regularization** is a technique used to reduce the error by fitting a function appropriately on the given training set and avoid overfitting. For a linear model, regularization is typically achieved by constraining the weights of the model. For example:

- The loss function of linear regression model:

```math
J(\theta) = MSE(\theta) = \frac{1}{m} \sum_{i=1}^{m} (\theta^T \cdot x_i - y_i)^2 =  \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
```

- We can add some regularization terms to the loss function to reduce the overfitting problem. The most common regularization techniques are **Ridge Regression**, **Lasso Regression**, and **Elastic Net**.

### **3.2. Ridge Regression**

- `Ridge regression` (also called Tikhonov regularization) is a regression model that uses `L2 regularization`.

- The cost function of Ridge Regression is as follows, the first is the MSE loss function, and the second is the regularization term. Note that the bias term $`\theta_0`$ is not regularized (the sum starts at i = 1, not 0).

```math
J(\theta) = MSE(\theta) + \alpha \frac{1}{2} \sum_{i=1}^{n} \theta_i^2
```

- The regularization term forces the learning algorithm to not only fit the data but also keep the model weights as small as possible.

- The hyperparameter $`\alpha`$ controls how much you want to regularize the model.
  - $`\alpha`$ = 0 then Ridge Regression is just Linear Regression.
  - $`\alpha`$ is very large, then all weights end up very close to zero and the result is a flat line going through the data’s mean.

- Ridge regression can be useful for linear regression for the following reasons:
  - `Reduce overfitting`. By shrinking the coefficients, ridge regression imposes a bias that limits the model's complexity and ability to fit the training data exactly. This improves the model's generalization ability for new data.
  - `Handle multicollinearity and Improve stability`. When features are highly correlated, multicollinearity can cause coefficients to have high variance and be sensitive to small changes in the data. Ridge regression stabilizes the coefficients by distributing predictive power across correlated features.
  - `Control variance and bias`. The $`\alpha`$ regularization hyperparameter allows control over the balance between variance (model complexity) and bias (similarity to true model). Increasing $`\alpha`$ increases the bias and decreases the variance, and vice versa.

### **3.3. Lasso Regression**

- `Lasso Regression` (Least Absolute Shrinkage and Selection Operator Regression) is a regression model that uses `L1 regularization`.

- The cost function of Lasso Regression is as follows:

```math
J(\theta) = MSE(\theta) + \alpha \sum_{i=1}^{n} |\theta_i|
```

- Lasso Regression automatically performs feature selection and outputs a sparse model (i.e., with few nonzero feature weights). In other words, it tends to eliminate the weights of the least important features (i.e., set them to zero).

- Why Lasso can be used for feature selection? Because when $`\alpha`$ is sufficiently large, many of the parameter estimates will be set to zero. This is because the L1 penalty is not differentiable at zero, and so the parameter estimates will be “pushed” towards zero as the penalty is increased.

- Lasso Regression can be useful for the following reasons:
  - Perform automatic feature selection by driving weak coefficients to exactly zero.
  - Work well for high-dimensional data by selecting a sparse set of relevant features.
  - Have less variance compared to Ridge regression, since it selects fewer features. But it has higher bias due to shrinking some coefficients to zero.
  - Control the balance between sparsity and goodness-of-fit via $`\alpha`$ regularization parameter.

### **3.4. Comparison between Ridge and Lasso**

| L1 Regularization | L2 Regularization |
|:--|:--|
| The penalty term is based on the absolute values of the model's parameters.| The penalty term is based on the squares of the model's parameters.|
| Produces sparse solutions (some parameters are shrunk towards zero).| Produces non-sparse solutions (all parameters are used by the model). |
| Sensitive to outliers.| Robust to outliers.|
| Selects a subset of the most important features.| All features are used by the model.|
| Optimization is non-convex.| Optimization is convex.|
| The penalty term is less sensitive to correlated features.| The penalty term is more sensitive to correlated features.|
| Useful when dealing with high-dimensional data with many correlated features.| Useful when dealing with high-dimensional data with many correlated features and when the goal is to have a less complex model.|
| Also known as Lasso regularization.| Also known as Ridge regularization.|

![L1vsL2](https://miro.medium.com/v2/resize:fit:720/1*etUdoAT3P51qjMiadDRHhQ.gif)

- We can see that with the L2 norm, as weight gets smaller so does the slope of the norm, meaning that the updates will also become smaller and smaller. When the weights are close to 0 the updates will have become so small as to be almost negligible, so it’s unlikely that the weights will ever become 0.

- On the other hand, with the L1 norm the slope is constant. This means that as weight gets smaller the updates don’t change, so we keep getting the same “reward” for making the weights smaller. Therefore, the L1 norm is much more likely to reduce some weights to 0.

- **Why Lasso can be used for feature selection?**
  - Assume that we have  a linear regression model with 2 features $`\beta_1`$ and $`\beta_2`$, so the cost function is:
    - Lasso: $`J(\beta_1, \beta_2) = MSE(\beta_1, \beta_2) + \alpha |\beta_1| + \alpha |\beta_2|`$.
    - Ridge: $`J(\beta_1, \beta_2) = MSE(\beta_1, \beta_2) + \alpha \beta_1^2 + \alpha \beta_2^2`$.
  - The specified domain of the cost function is a circular for Ridge and a diamond for Lasso.
  - The elliptical contours (red circles) are the cost functions for each.

![L1vsL2](https://images.datacamp.com/image/upload/v1648205672/image18_a3zz7y.png)

- The intersection of the cost function and the domain is the optimal solution.
  - Lasso takes a diamond shape, each time the elliptical regions intersect with these corners, at least one of the coefficients becomes zero.
  - Ridge forms a circular shape and therefore values can be shrunk close to zero, but never equal to zero.

### **3.4. Elastic Net**

- `Elastic Net` is a regression model that combines both L1 and L2 regularization.

- The cost function of Elastic Net is as follows:

```math
J(\theta) = MSE(\theta) + r \alpha \sum_{i=1}^{n} |\theta_i| + \frac{1-r}{2} \alpha \sum_{i=1}^{n} \theta_i^2
```

- Elastic Net is a middle ground between Ridge Regression and Lasso Regression. The regularization term is a simple mix of both Ridge and Lasso’s regularization terms, and we can control the mix ratio `r`.
  - `r = 0`, Elastic Net is equivalent to Ridge Regression.
  - `r = 1`, it is equivalent to Lasso Regression.

### **3.5. Early Stopping**

- A very different way to regularize iterative learning algorithms such as Gradient Descent is to stop training as soon as the validation error reaches a minimum. This is called early stopping.

![Early Stopping](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https:%2F%2Fblog.kakaocdn.net%2Fdn%2Fd3sl2A%2FbtqDvYdRCkM%2FAn0fCdvjk6MLUUSLYk4dKK%2Fimg.png)

- In the above figure:
  - As the epochs go by the algorithm learns, and its prediction error (RMSE) on the training set goes down, along with its prediction error on the validation set.
  - After a while, the validation error stops decreasing and starts to go back up. This indicates that the model has started to overfit the training data.
  - With early stopping we just stop training as soon as the validation error reaches the minimum.
