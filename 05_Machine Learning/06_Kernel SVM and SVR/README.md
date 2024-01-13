# **MACHINE LEARNING PART 3**

## **3. Kernel Support Vector Machine**

### **3.1. Introduction**

- Kernel SVM is a non-linear SVM applied for non-linear data, It uses a kernel function to transform the input data into a higher dimensional space, where the data is linear separable.

- Let's consider the following example:

||||
|--|--|--|
|![Kernel](https://machinelearningcoban.com/assets/21_kernelsvm/5.png) | ![Kernel](https://machinelearningcoban.com/assets/21_kernelsvm/4.png) | ![Kernel](https://machinelearningcoban.com/assets/21_kernelsvm/6.png) |

- In the figure:
  - From non-linear data in 2D space, we use a kernel function to transform the data into a 3D space, so that the data is linear separable.
  - The third dimension can be calculated by the following formula: $`z = x^2 + y^2`$.
  - The yellow plane is the decision boundary in 3D space, it's can be found by SVM.
  - The intersection of the found plane and the parabolic surface is an ellipse, when projecting into the original 2D space, we have found the line dividing the two classes.

- **In short:** Kernel SVM is to find a function to transform the data into a higher dimensional space, so that the data is linear separable. This function is called a kernel function.

### **3.2. Mathematical foundations**

- The following is short solution of SVM problem using Lagrange multiplier. The class of a data point can be calculated by the sign of the following formula:

```math
h_{\mathbf{w}, b}(\mathbf{x}_i) = b + \sum_{j\in \mathcal{S}} \lambda_j y_j \mathbf{x}_{j}^{\intercal} \mathbf{x}_i
```

- In which:
  - $`\lambda_n`$ is the Lagrange multiplier of the n-th data point.
  - $`\mathcal{S}`$ is the set of data points that $`\lambda_n > 0`$.

- In the case of nonlinear transformation we consider in the previous section, we have:

```math
h_{\mathbf{w}, b}(\Phi(\mathbf{x}_i))  =  b + \sum_{j\in \mathcal{S}} \lambda_j y_j \Phi(\mathbf{x}_{j})^{\intercal} \Phi(\mathbf{x}_i) = b + \sum_{j\in \mathcal{S}} \lambda_j y_j k(\mathbf{x}_{j}, \mathbf{x}_i)
```

- In which:
  - $`\Phi(\mathbf{x})`$ is the nonlinear transformation function.
  - $`k(\mathbf{x}_{j}, \mathbf{x}_i) = \Phi(\mathbf{x}_{j})^{\intercal} \Phi(\mathbf{x}_i)`$ is the kernel function.

- We can see that, instead of calculating the nonlinear transformation function, we can find a kernel function $`k(\mathbf{x}_{j}, \mathbf{x}_i) = \phi(\mathbf{x}_{j})^{\intercal} \phi(\mathbf{x}_i)`$ and calculate the kernel function directly. (This is called the `kernel trick`).

- Consider a example: we transform 1 data point in 2D space $`\mathbf{x} = [x_1, x_2]^T`$ into a point in 5D space $`\Phi(\mathbf{x}) = [1, \sqrt{2} x_1, \sqrt{2} x_2, x_1^2, \sqrt{2} x_1x_2, x_2^2]^T`$. We have:

```math
\begin{matrix}
\Phi(\mathbf{x})^T\Phi(\mathbf{z}) &=& [1, \sqrt{2} x_1, \sqrt{2} x_2, x_1^2, \sqrt{2} x_1x_2, x_2^2] [1, \sqrt{2} z_1, \sqrt{2} z_2, z_1^2, \sqrt{2} z_1z_2, z_2^2]^T \newline
\newline
&=& 1 + 2x_1z_1 + 2x_2z_2 + x_1^2x_2^2 + 2x_1z_1x_2z_2 + x_2^2z_2^2 \newline
\newline
&=& (1 + x_1z_1 + x_2z_2)^2 = (1 + \mathbf{x}^T\mathbf{z})^2 = k(\mathbf{x}, \mathbf{z})
\end{matrix}
```

- We can see that it's more easy to calculate the kernel function than calculate the nonlinear transformation function and then calculate the dot product of the two vectors.

### **3.3. Kernel of SVM**

|Kernel|Formula|sklearn parameter|Kernel parameter|
|--|--|--|--|
|Linear| $`\mathbf{x}^T\mathbf{z}`$ | `kernel='linear'` | C |
|Polynomial| $`(r + \gamma \mathbf{x}^T\mathbf{z})^d`$ | `kernel='poly'` | C, $`\gamma`$, d|
|Sigmoid| $`\text{tanh}(\gamma \mathbf{x}^T\mathbf{z} + r)`$ | `kernel='sigmoid'` | C, $`\gamma`$|
|Gaussian| $`exp(-\gamma \|\|\mathbf{x} - \mathbf{z}\|\|_2^2)`$ | `kernel='rbf'` | C, $`\gamma`$ |

- The 2 most important parameters of SVM are $`C`$ and $`\gamma`$:
  - C
    - Tell the algorithm how much to avoid misclassifying each training sample.
    - The larger C, the smaller the margin, and, conversely, the smaller C, the larger the margin.
    - Can be used along with any kernel.
![C](https://s3.stackabuse.com/media/articles/understanding-svm-hyperparameters-1.png)

  - $`\gamma`$
    - Define how far influences the calculation of plausible line of separation.
    - When gamma is higher, nearby points will have high influence; low gamma means far away points also be considered to get the decision boundary.
![Gamma](https://miro.medium.com/max/1713/1*6HVomcqW7BWuZ2vvGOEptw.png)

## **4. Support Vector Regression (SVR)**

### **4.1. What is SVR?**

- **Support Vector Regression (SVR)** is a type of machine learning algorithm used for regression analysis.

- Unlike SVMs used for classification tasks, SVR seeks to find a hyperplane that best fits the data points in a continuous space.

- SVR can handle non-linear relationships between the input variables and the target variable by using a kernel function to map the data to a higher-dimensional space.

### **4.2. How does SVR work?**

- The problem of regression is to find a function that approximates mapping from an input domain to real numbers on the basis of a training sample.

![SVR](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/03/SVR1.png)

- Consider these two red lines as the decision boundary and the green line as the hyperplane. Our objective is to basically consider the points that are within the decision boundary line. Our best fit line is the hyperplane that has a maximum number of points.

![SVR](https://miro.medium.com/v2/resize:fit:720/format:webp/1*rs0EfF8RPVpgA-EfgAq85g.jpeg)

- Assuming that the equation of the hyperplane is as follows:

```math
Y = wx+b
```

- Then the equations of decision boundary become:

```math
wx+b= +\epsilon
\newline
wx+b= -\epsilon
```

- Our main aim here is:
  - Decide a decision boundary at $`\epsilon`$ distance from the original hyperplane such that data points **closest to the hyperplane** (within that boundary line).

  - We take only those points that are within the decision boundary and have the least error rate (or accepted Tolerance). This gives us a better fitting model.

- Unlike other Regression models that try to minimize the error between the real and predicted value, the SVR tries to fit the best line within a threshold  (epsilon). The threshold value is the distance between the hyperplane and boundary line.

|||
|---|---|
| ![SVR](https://www.saedsayad.com/images/SVR_1.png) | ![SVR](https://www.saedsayad.com/images/SVR_2.png) |

### **4.5. SVR advantages and disadvantages**

- Advantages of SVR
  - SVR have good generalization capability
  - SVR is more effective in high dimensional spaces.
  - SVR can work well even if the data is not linearly separable.

- Disadvantages of Support Vector Regression
  - Slow training time so it's not suitable for large datasets.
  - Difficulty in choosing appropriate hyperparameters.
  - Potential for overfitting if hyperparameters are not tuned well.
