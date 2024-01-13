# **MACHINE LEARNING PART 3**

## **1. Support Vector Machine (SVM)**

### **1.1. Distance from a point to a plane**

- In d dimensions space, distance from a point (vector) have coordinates $`(x_1, x_2, ..., x_d)`$ to a plane have equation $`w_1x_1 + w_2x_2 + ... + w_dx_d + b = 0`$ is calculated by:

```math
d = \frac{|w_1x_1 + w_2x_2 + ... + w_dx_d + b|}{\sqrt{w_1^2 + w_2^2 + ... + w_d^2}}
```

- General, with a point (vector) $`\mathbf{x}_i`$ and a plane $`w^T\mathbf{x} + b = 0`$, we have:

```math
d = \frac{|\mathbf{w}^T\mathbf{x}_i + b|}{||\mathbf{w}||_2}
```

- In which, $`||\mathbf{w}||_2 = \sqrt{\sum_{i=1}^d w_i^2}`$ is the norm of vector $`\mathbf{w}`$ in $`d`$ dimensions space.

### **1.2. Objective of SVM**

- The objective of SVM is to find a hyperplane in an d-dimensional space that separate the data into two classes. The best hyperplane is the one that have the largest distance to the nearest training data point of any class (this hyperplane maximizes the margin between the two classes). That is why SVM is also known as the Large Margin Classifier.

|||
|--|--|
|![Large margin](https://miro.medium.com/v2/resize:fit:600/format:webp/0*9jEWNXTAao7phK-5.png) | ![Large margin](https://miro.medium.com/v2/resize:fit:600/format:webp/0*0o8xIA4k3gXUDCFU.png) |

### **1.3.Optimization SVM**

- Asume we have the following training set:
  - $`(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_N, y_N)`$ is data points.
  - Vector $`\mathbf{x}_i \in \mathbb{R}^d`$ is the i-th input data point.
  - $`y_i \in \{-1, 1\}`$ is the label of i-th data point.
  - $`N`$ is number of data points.
  - $`d`$ is number of dimensions of data points (number of features).

|||
|--|--|
|![Analyze SVM](https://machinelearningcoban.com/assets/19_svm/svm6.png) |![Analyze SVM](https://machinelearningcoban.com/assets/19_svm/svm3.png) |

- In the above figure:
  - The blue squares point is data points with label $`y_i = 1`$.
  - The red circles point is data points with label $`y_i = -1`$.
  - $`\mathbf{w}^T\mathbf{x} + b = w_1x_1 + w_2x_2 + b = 0`$ is the hyperplane that separate the data points into two classes.
  - We asume that class 1 is in positive side of the hyperplane and class -1 is in negative side of the hyperplane.

- With a random pair $`(\mathbf{x}_n, y_n)`$, the distance from that point to the hyperplane is:

```math
\frac{y_n(\mathbf{w}^T\mathbf{x}_n + b)}{||\mathbf{w}||_2}
```

- Why $`y_n(\mathbf{w}^T\mathbf{x}_n + b)`$?
  - Follow the above asumption, $`y_n`$ will always same sign with $`x_n`$, so $`y_n(\mathbf{w}^T\mathbf{x}_n + b)`$ will always positive. (If $`y_n = 1`$, $`x_n`$ will be in positive side of the hyperplane, so $`\mathbf{w}^T\mathbf{x}_n + b`$ will be positive. If $`y_n = -1`$, $`x_n`$ will be in negative side of the hyperplane, so $`\mathbf{w}^T\mathbf{x}_n + b`$ will be negative.)

- With the above figure, `margin` will be the the smallest distance from a point to the hyperplane. So, we have:

```math
\text{margin} = \min_{n} \frac{y_n(\mathbf{w}^T\mathbf{x}_n + b)}{||\mathbf{w}||_2}
```

- Our goal is to find the parameters $`(\mathbf{w}, b)`$ that make the hyperplane separate the data points into two classes and max the above `margin`.

```math
(\mathbf{w}, b) = \arg\max_{\mathbf{w}, b} \left\{
    \min_{n} \frac{y_n(\mathbf{w}^T\mathbf{x}_n + b)}{||\mathbf{w}||_2} 
\right\}
= \arg\max_{\mathbf{w}, b}\left\{
    \frac{1}{||\mathbf{w}||_2} \min_{n} y_n(\mathbf{w}^T\mathbf{x}_n + b)
\right\} ~~~ (1)
```

- We have a comment that if we scale the parameters $`(\mathbf{w}, b)`$ by a constant positive number `k`, the hyperplane will not change (the distance from data points to hyperplane not change). Base on that, we can asume the following for the points that stay nearest the hyperplane:

```math
y_n(\mathbf{w}^T\mathbf{x}_n + b) = 1
```

- So with all data points, we have:

```math
y_n(\mathbf{w}^T\mathbf{x}_n + b) \geq 1, \forall n = 1, 2, \dots, N
```

- We can rewrite the equation (1) as:

```math
\begin{matrix}
    (\mathbf{w}, b) = \arg \max_{\mathbf{w}, b} \frac{1}{||\mathbf{w}||_2}   \newline
    \newline
    \text{subject to:}~ y_n(\mathbf{w}^T\mathbf{x}_n + b) \geq 1, \forall n = 1, 2, \dots, N ~~~~(2)
\end{matrix}
```

- Take inverse of the equation (2), we have:

```math
\begin{matrix}
    (\mathbf{w}, b) = \arg \min_{\mathbf{w}, b} \frac{1}{2}||\mathbf{w}||_2^2   \newline
    \newline
    \text{subject to:}~ 1 - y_n(\mathbf{w}^T\mathbf{x}_n + b) \leq 0, \forall n = 1, 2, \dots, N ~~~~ (3)
\end{matrix}
```

- Here, we take the inverse of equation (2), then square the norm of $`\mathbf{w}`$ and multiply by $`\frac{1}{2}`$ to make the objective function in equation (3) is differentiable. (The objective function in equation (2) is not differentiable because of the norm.)

#### **Important observation**

- The problem in equation (3) is a convex optimization problem.
- It's very complicated to solve the problem in equation (3) directly. So, we will use the Lagrange multiplier method to solve it.
- Detail of how to solve the problem in equation (3) by Lagrange multiplier method can be found [here](https://machinelearningcoban.com/2017/04/09/smv/) and [here](https://phamdinhkhanh.github.io/deepai-book/ch_ml/SVM.html).

#### **Define class for a new data point**

- After solve the above problem and find the optimal parameters $`(\mathbf{w}, b)`$, we have the hyperplane $`\mathbf{w}^T\mathbf{x} + b = 0`$. For a new data point $`\mathbf{x}_i`$, we can define the class of that point by:

```math
\text{class}(\mathbf{x}) = \text{sign} (\mathbf{w}^T\mathbf{x} + b )
```

- `sign` returns 1 if the input is positive, -1 otherwise.

## **2. Soft Margin Support Vector Machine**

### **2.1. Problem**

- The above SVM is a `Hard Margin SVM`, it only works well when the data is linearly separable. But in reality, the data is not always linearly separable. Naturally, we also wish that SVM could work with data as near linearly separable as Logistic Regression did.

- Let's consider the following figure:

|||
|--|--|
|![Soft Margin SVM](https://machinelearningcoban.com/assets/20_softmarginsvm/ssvm1.png)|![Soft Margin SVM](https://machinelearningcoban.com/assets/20_softmarginsvm/ssvm2.png)|

- Two cases of data that Hard Margin SVM can't work well:
  - Noise data points: the margin will be very small. So, Hard Margin SVM is very sensitive to outliers.
  - Not linearly separable but near linearly separable. In this case, Hard Margin SVM can't find the hyperplane that separate the data points into 2 classes.

- Solution: allow some data points to be on the wrong side of the hyperplane, we can find the hyperplane that separate the data points into 2 classes. We call this method `Soft Margin SVM`.

- For Soft Margin SVM, we have 2 different approaches to solve the problem:
  - *The first approach* is to solve a constrained optimization problem, we will add a slack variable $`\xi_n`$ for each data point $`\mathbf{x}_n`$ to the problem in equation (3) of Hard Margin SVM.
  - *The second approach* is to solve an unconstrained optimization problem, this approach can be solved by Graident Descent method. We will use a new loss function called `Hinge Loss`.

### **2.2. First approach**

- We will `sacrifice` some data points by allowing them to fall into the `unsafe zone`. Of course, we must limit this `sacrifice`, if not the hyperplane will be very bad.

- So the objective function of Soft Margin SVM must be a combination of:
  - Maximizing the margin
  - Minimizing the number of points that fall into the unsafe zone.

- Like Hard Margin SVM, we can maximize the margin by minimizing $`||\mathbf{w}||_2^2`$.

- To determine the number of points that fall into the unsafe zone, let's consider the following figure:

![Unsafe zone](https://machinelearningcoban.com/assets/20_softmarginsvm/ssvm3.png)

- In the above figure, we have:
  - New variable $`\xi_n`$ called `slack variable` to calculate the 'sacrifice' for each data point $`\mathbf{x}_n`$.
    - With points in safe zone, $`\xi_n = 0`$.
    - With points in unsafe zone but on the correct side of the hyperplane, we have $`0 < \xi_n < 1`$. E.g: $`\mathbf{x}_2`$.
    - With points in unsafe zone and on the wrong side of the hyperplane, we have $`\xi_n > 1`$. E.g: $`\mathbf{x}_1, \mathbf{x}_3`$

- We have that if $`y_i = \pm 1`$ is the label of $`\mathbf{x}_i`$ in the unsafe zone, we can define the `slack variable` $`\xi_i`$ as:

```math
\xi_i = |\mathbf{w}^T\mathbf{x}_i + b - y_i|
```

- We have the optimization problem of Hard Margin SVM is:

```math
\begin{matrix}
    (\mathbf{w}, b) = \arg \min_{\mathbf{w}, b} \frac{1}{2}||\mathbf{w}||_2^2   \newline
    \newline
    \text{subject to:}~ 1 - y_n(\mathbf{w}^T\mathbf{x}_n + b) \leq 0, \forall n = 1, 2, \dots, N ~~~~ (3)
\end{matrix}
```

- In Soft Margin SVM, we will add a new term to limit the `sacrifice`, so **the objective function** of Soft Margin SVM is:

  ```math
  \frac{1}{2}{||\mathbf{w}||_2^2} + C \sum_{n=1}^N \xi_n
  ```

  - In which, C is a positive constant that we can choose and $`\xi = [\xi_1, \xi_2, \dots, \xi_N]`$ is the vector of slack variables.

- **Soft binding condition**:

  ```math
  y_n(\mathbf{w}^T\mathbf{x}_n + b) \geq 1 - \xi_n \Leftrightarrow 1 - \xi_n - y_n(\mathbf{w}^T\mathbf{x}_n + b) \leq 0, ~~ \forall n = 1, 2, \dots, n
  ```
  
  - And we have a sub-binding condition: $`\xi_n \geq 0, ~\forall n = 1, 2, \dots, N`$.

#### **Conclusion**

- So, we have the optimization problem of Soft Margin SVM is:

```math
\begin{matrix}
    (\mathbf{w}, b, \xi) = \arg \min_{\mathbf{w}, b, \xi} \frac{1}{2}{||\mathbf{w}||_2^2} + C \sum_{n=1}^N \xi_n  \newline
    \newline
    \text{subject to:}~  1 - \xi_n - y_n(\mathbf{w}^T\mathbf{x}_n + b) \leq 0, \forall n = 1, 2, \dots, N \newline
    \newline
    -\xi_n \leq 0,  ~\forall n = 1, 2, \dots, N
\end{matrix}
```

- The optimization problem of Soft Margin SVM is convex. So we can use `Lagrange Multiplier` to solve this problem. `This is the First Approach`.

- We have some comments:
  - If C is small, the model will tend to maximize the margin. That leads to the fact $`\sum_{n=1}^N\xi_n`$ get larger. So when C get larger, the margin get smaller.
  - Otherwise, if C is large, to minimize the objective function, model will try to make $`\sum_{n=1}^N\xi_n`$ smaller. If C is pretty large and the data is linearly separable, we have $`\sum_{n=1}^N\xi_n = 0`$, that mean no data point fall into the unsafe zone and the model will be the same as Hard Margin SVM.

  |||
  |--|--|
  |![C1](https://machinelearningcoban.com/assets/20_softmarginsvm/ssvm5_01.png) |  ![C2](https://machinelearningcoban.com/assets/20_softmarginsvm/ssvm5_1.png) |
  |![C3](https://machinelearningcoban.com/assets/20_softmarginsvm/ssvm5_10.png) |  ![C4](https://machinelearningcoban.com/assets/20_softmarginsvm/ssvm5_100.png) |

### **2.3. Second approach**

#### **2.3.1. Equivalence unconstrained optimization problem**

- Let's check the first binding condition:

```math
1 - \xi_n -y_n(\mathbf{w}^T\mathbf{x} + b)) \leq 0 \Leftrightarrow \xi_n \geq 1 - y_n(\mathbf{w}^T\mathbf{x} + b))
```

- We have that $`\xi_n \geq 0`$ . So we can rewrite the optimization problem of Soft Margin SVM as:

```math
\begin{matrix}
    (\mathbf{w}, b, \xi) = \arg \min_{\mathbf{w}, b, \xi} \frac{1}{2}{||\mathbf{w}||_2^2} + C \sum_{n=1}^N \xi_n  \newline
    \newline
    \text{subject to:}~ \xi_n \geq \max(0, 1 - y_n(\mathbf{w}^T\mathbf{x} + b)), ~\forall n = 1, 2, \dots, N
\end{matrix}
```

- We have that when $`(\mathbf{w}, b, \xi)`$ is the solution of the above problem, i.e. where the objective function reaches its minimum value, then:

  ```math
  \xi_n = \max(0, 1 - y_n(\mathbf{w}^T\mathbf{x}_n + b)), ~\forall n = 1, 2, \dots, N
  ```

- We replace all the value of $`\xi_n`$ by $`\max(0, 1 - y_n(\mathbf{w}^T\mathbf{x}_n + b))`$ in the objective function, we have:

```math
\begin{matrix}
    (\mathbf{w}, b, \xi) = \arg \min_{\mathbf{w}, b, \xi} \frac{1}{2}{||\mathbf{w}||_2^2} + C \sum_{n=1}^N \max(0, 1 - y_n(\mathbf{w}^T\mathbf{x}_n + b)) \newline
    \newline
    \text{subject to:}~ \xi_n = \max(0, 1 - y_n(\mathbf{w}^T\mathbf{x}_n + b)), ~\forall n = 1, 2, \dots, N
\end{matrix}
```

- Obviously the slack variable $`\xi_n`$ is no longer needed. So we have the optimization problem of Soft Margin SVM is:

```math
(\mathbf{w}, b)= \arg \min_{\mathbf{w}, b} \frac{1}{2}{||\mathbf{w}||_2^2} + C \sum_{n=1}^N \max(0, 1 - y_n(\mathbf{w}^T\mathbf{x}_n + b)) \triangleq \arg\min_{\mathbf{w}, b} J(\mathbf{w}, b)
```

- This is the unconstrained optimization problem. We can use `Gradient Descent` to solve this problem. `This is the Second Approach`.

#### **2.3.2. Hinge loss function**

- We have a new loss function called `Hinge loss function`:

```math
J_n(\mathbf{w}, b) = \max(0, 1 - y_nz_n)
```

- In which, $`z_n = \mathbf{w}^T\mathbf{x}_n + b`$ is the score of $`\mathbf{x}_n`$ corresponding to $`(\mathbf{w}, b)`$. $`y_n`$ is the desired output.

![Hinge loss](https://miro.medium.com/v2/resize:fit:1150/1*PGqpYm7o5GCbDXxXErr2JA.png)

- With Hinge Loss, we have:
  - With points in safe zone ($`y_nz_n > 1`$), $`J_n(\mathbf{w}, b) = 0`$. Cause no error.
  - With points in unsafe zone but classified correctly, so ($`0 < y_nz_n < 1`$), $`J_n(\mathbf{w}, b) = 1 - y_nz_n`$. Cause a small error.
  - With points in unsafe zone but misclassified, so ($`y_nz_n < 0`$), $`J_n(\mathbf{w}, b) = 1`$. Cause a big error.

- Hinge loss is also a continuous function and almost everywhere differentiable (a part from the point $`y_nz_n = 1`$). And it's derivative is easy to determine: equal to -1 if $`y_nz_n < 1`$ and equal to 0 if $`y_nz_n > 1`$. At the point $`y_nz_n = 1`$, the derivative is not defined, but we can take it as 0.

#### **2.3.3. Construct Loss function**

- With each pair $`(\mathbf{w}, b)`$, we assume:

```math
L_n(\mathbf{w}, b) = \max(0, 1 - y_n(\mathbf{w}^T\mathbf{x}_n + b))
```

- So with all training set, we have the loss:

```math
L(\mathbf{w}, b) = \sum_{n=1}^N L_i = \sum_{n=1}^N \max(0, 1 - y_n(\mathbf{w}^T\mathbf{x}_n + b))
```

- If we optimize directly the above loss function, there will be some problems:
  - In case data is linear separable, so the optimized value of $`L(\mathbf{w}, b) = 0`$, that means:

  ```math
  1 - y_n (\mathbf{w}^T\mathbf{x}_n + b) \leq 0, ~\forall n = 1, 2, \dots, N
  ```

  - Multiply both side by a constant a > 1:

  ```math
  \begin{matrix}
  a - y_n (a\mathbf{w}^T\mathbf{x}_n + ab) &\leq& 0, ~\forall n = 1, 2, \dots, N \newline
  \newline
  \Rightarrow 1 - y_n (a\mathbf{w}^T\mathbf{x}_n + ab) &\leq& 1 - a < 0, ~\forall n = 1, 2, \dots, N
  \end{matrix}
  ```

  - That mean $`(a\mathbf{w}, ab)`$ is also a solution of the optimization problem. So the solution is not unique.

- To avoid this problem, we add a regularization term to the loss function, here we use L2 regularization:

  ```math
  J(\mathbf{w}, b) = \sum_{n=1}^N \max(0, 1 - y_n(\mathbf{w}^T\mathbf{x}_n + b)) + \frac{\lambda}{2} ||\mathbf{w}||_2^2
  ```

  - In which $`\lambda > 0`$ is the regularization parameter.

- The above technique is called `weight decay`. Note that `weight decay` is not applied to the bias term $`b`$.

- To simpler, we can use a `bias trick`:
  - We add a new dimension (add a new component equal to 1) to the data, so that $`\mathbf{x}_n \in \mathbb{R}^{d+1}`$ and $`\mathbf{w} \in \mathbb{R}^{d+1}`$.
  - We can combine $`\mathbf{w}`$ and $`b`$ into a new vector $`\bar{\mathbf{w}} = [\mathbf{w}^T, b]^T \in \mathbb{R}^{d+1}`$.

- Now, we have a simpler loss function:

```math
J(\mathbf{\bar{w}}) = \underbrace{\sum_{n=1}^N \max(0, 1 - y_n\bar{\mathbf{w}}^T\mathbf{\bar{x}}_n)}_{\text{hinge loss}} + \underbrace{\frac{\lambda}{2} ||\mathbf{w}||_2^2}_{\text{regularization}}
```

- We can see that the problem becomes unconstraint, so we can use `Gradient Descent` to solve this problem. The loss function is also a convex function, so with a suitable learning rate and iteration, we can find the global minimum of the loss function.

#### **2.3.4. Optimize Loss function**

- With regularization term, we have its derivative by $`\mathbf{\bar{w}}`$ is:

  ```math
  \lambda \left[\begin{matrix}
  \mathbf{w}\newline
  0
  \end{matrix}\right]
  ```

  - 0 is the derivative of $`b`$ in regularization term.

- With Hinge loss, consider each data point, we have 2 case:
  - Case 1: If $`1 - y_n \mathbf{\bar{w}}^T\mathbf{\bar{x}}_n \leq 0`$, the derivative of Hinge loss by $`\mathbf{\bar{w}}`$ is 0.
  - Case 2: If $`1 - y_n \mathbf{\bar{w}}^T\mathbf{\bar{x}}_n > 0`$, the derivative of Hinge loss by $`\mathbf{\bar{w}}`$ is $`-y_n \mathbf{\bar{x}}_n`$.

- For case 2, take a vector `u`:

```math
\mathbf{u} = [y_1\mathbf{\bar{w}}^T\mathbf{\bar{x}}_1,y_2\mathbf{\bar{w}}^T\mathbf{\bar{x}}_2, \dots, y_N \mathbf{\bar{w}}^T \mathbf{\bar{x}}_N]
```

- We find all the point that `u < 1`, by setting a set:

```math
\mathcal{H} = \{n: u_n < 1\}
```

- Finally, we have the derivative of loss function by $`\mathbf{\bar{w}}`$ is:

```math
\nabla J(\mathbf{\bar{w}}) = \sum_{n \in \mathcal{H}} - y_n\mathbf{\bar{x}}_n  + \lambda 
\left[\begin{matrix}
\mathbf{w}\newline
0
\end{matrix}\right]
```

- Using Gradient Descent, we have update equation:
  
```math
\mathbf{\bar{w}} = \mathbf{\bar{w}} - \eta \left(\sum_{n \in \mathcal{H}} - y_n\mathbf{\bar{x}}_n  + \lambda \left[\begin{matrix}
\mathbf{w}\newline
0
\end{matrix}\right]\right)
```
