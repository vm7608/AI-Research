# **MACHINE LEARNING PART 2**

## **2. Gradient Descent**

### **2.1. Introduction**

- Gradient Descent is a generic optimization algorithm capable of finding optimal solutions to a wide range of problems. The general idea of Gradient Descent is to tweak parameters iteratively in order to minimize a cost function.

- General steps of Gradient Descent algorithm:
  - `Step 1:` Randomly initialize the parameters ($`\theta`$)

  - `Step 2:` Calculate the gradient of the cost function at the current point. This value is called the *gradient vector*.

  - `Step 3:` Move in the direction of the negative gradient to minimize the cost function. Update the parameters ($`\theta`$) based on the gradient of the cost function. The size of the step is determined by the *learning rate*. The equation to update the parameters is:

    ```math
    \theta_{i+1} = \theta_i - \eta \nabla J(\theta_i)
    ```

    - $`\theta_i`$ is the $`i_{th}`$ parameter, it's can be the weights or the bias.
    - $`\eta`$ is the learning rate.
    - $`\nabla J(\theta_i)`$ is the derivative of the cost function at the $`i_{th}`$ parameter ($`\theta_i`$).
  
  - `Step 4:` Repeat the process until the result reaches an aceptable value or the maximum number of iterations is reached.

![Gradient Descent](https://blog.kakaocdn.net/dn/MKRA8/btrAPf2Fil6/s4KoiMl0qEVqTzpWvj9x6K/img.gif)
*Example of Gradient Descent in 2D space ax+b*

### **2.2. Initialization point and learning rate**

#### **2.2.1. Initialization point**

- The `initialization point` is the starting point of the algorithm. It is usually chosen randomly.

- The initialization point is a hyperparameter, choosing it is challenging as a poor choice may result in a sub-optimal set of weights or a slow convergence of the learning algorithm.

![Initialization point](https://machinelearningcoban.com/assets/GD/1dimg_5_0.1_-5.gif)
![Initialization point](https://machinelearningcoban.com/assets/GD/1dimg_5_0.1_5.gif)

- In the above example:
  - The upper figure shows the initialization point is -5 and the lower figure shows the initialization point is 5.
  - With the same learning rate, the algorithm converges faster when the initialization point is -5.

#### **2.2.2. Learning rate**

- The `learning rate` is the size of the step that the algorithm takes in the direction of the negative gradient. It is usually a small positive number.

- The learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.

- Choosing the learning rate is challenging as:
  - Too small value may result in a long training process that could get stuck.
  - Too large value may result in learning a sub-optimal set of weights too fast or an unstable training process.

![Learning rate](https://machinelearningcoban.com/assets/GD/1dimg_5_0.01_-5.gif)
![Learning rate](https://machinelearningcoban.com/assets/GD/1dimg_5_0.5_-5.gif)

- In the above example, with the same initialization point:
  - With a small learning rate = 0.01, the algorithm converges slowly. With a limited number of iterations, the algorithm may not reach the minimum point.
  - With a larger learning rate = 0.5, the algorithm converges faster. However, the algorithm does not reach the minimum point because the learning rate is too large.

![Learning rate](https://ndquy.github.io/assets/img/blog/1_ShhdswkZTInut3L6Nbbw3Q.png)

#### **2.2.3. Stopping criteria**

- Another hyperparameter is the `stopping criteria`. It is used to determine when the algorithm has converged to a minimum point and can stop iterating.

- There are some common stopping criteria:
  - The maximum number of iterations is reached.
  - The difference between the previous and current cost function is smaller than a threshold. This threshold is called the `tolerance`, a very small positive number.

#### **2.2.4. Gradient Descent pitfalls**

- In reality, not all cost function look like nice as regular bowls. There may be holes, ridges, plateaus, and all sorts of irregular terrains, making convergence to the minimum very difficult.

- Some pitfalls of Gradient Descent are:
  - *Convergence Rate*: GD can be very slow to converge.
  - *Local minimum*: GD can converge to the local minimum. It can not find the global minimum.
  - *Learning rate and initialization point selection*
  - *Saddle point*: GD can get stuck at a saddle point. A saddle point is a specific type of critical point (where the gradient is zero) that is neither a local minimum nor a local maximum.
  - *Plateau*: GD can get stuck at a plateau. A plateau is a flat region of the error surface where the gradient is close to zero.

![Gradient Descent pitfalls](https://miro.medium.com/v2/resize:fit:1400/1*sbJO1Dz-dXKFMKsyaQ8pvQ.png)

![Saddle point](https://upload.wikimedia.org/wikipedia/commons/4/42/Minima_and_Saddle_Point.png)

### **2.2 Batch Gradient Descent**

- `Batch Gradient Descent (BGD)` updates the model’s parameters using the gradient of the entire training set (computes the gradient using the whole dataset in one epoch).

- One cycle through entire training datasets is called a `training epoch`.

- BGD follow the steps below:
  - `Step 1`: Randomly initialize the parameters ($`\theta`$)
  - `Step 2`: Calculate the gradient of the cost function based `on the whole dataset`.
    - Loop through the whole dataset, pass each record into the model to get the predicted value.
    - Calculate the error between the predicted value and the actual value.
    - After looping through the *whole dataset*, calculate the average error.
    - Calculate the gradient of the cost function based on the average error.
  - `Step 3`: Move in the direction of the negative gradient to minimize the cost function. Update the parameters ($`\theta`$) based on the gradient of the cost function.
    - $`\theta_{i+1} = \theta_i - \eta \nabla J(\theta_i)`$
  - `Step 4`: Repeat the process until the result reaches an aceptable value or the maximum number of iterations is reached.

- For example with Linear Regression, we have cost function is MSE, so the gradient of the cost function is:

  ```math
  \frac{\delta}{\delta\theta_j}MSE(\theta)  = \frac{1}{m}\sum_{i=1}^m(\hat{y}_i - y_i)x^{(i)}_j
  ```

- So we have a more clearly form of the update equation using BGD for Linear Regression is:

  ```math
  \theta_{j+1} = \theta_j - \eta \frac{1}{m}\sum_{i=1}^m(\hat{y}_i - y_i)x^{(i)}_j
  ```

- **Advantages:**
  - Prevents bias in the parameter update because of the use of the whole training dataset.
  - Simple to implement. Easy to understand.
  - Less parameters to tune. (Only the learning rate.)

- **Disadvantages:**
  - Computationally expensive for large datasets.
  - Cannot be used for online learning: hard to update the model online as new training data arrives.

### **2.3 Stochastic Gradient Descent**

- `Stochastic Gradient Descent (SGD)` calculates the error and updates the model for each randomly selected training sample (computes the gradient using a single sample).

- SGD computes the gradient using a single sample. As a result, it is much faster than BGD when the dataset is large.

- SGD follow the steps below:
  - `Step 1`: Randomly initialize the parameters ($`\theta`$)
  - `Step 2`: Randomly select a sample from the training dataset.
    - In each iteration, only a single training sample is used.
    - The samples are selected randomly, in a random shuffle order.
  - `Step 3`: Calculate the gradient of the cost function based on the selected sample.
    - Only the loss and gradients for the single selected sample are calculated.
    - This is in contrast to BGD which calculates the average loss and gradients over the entire batch.
  - `Step 4`: Move in the direction of the negative gradient to minimize the cost function. Update the parameters ($`\theta`$) based on the gradient of the cost function.
    - $`\theta_{i+1} = \theta_i - \eta \nabla J(\theta_i; \mathbf{x}_i; \mathbf{y}_i)`$
  - `Step 5`: Repeat the process until the result reaches an aceptable value or the maximum number of iterations is reached.

- Still with Linear Regression, we have update equation using SGD for Linear Regression is:

  ```math
  \theta_{j+1} = \theta_j - \eta (\hat{y}_i - y_i)x^{(i)}_j
  ```

- **Advantages:**
  - Computationally efficient for large datasets.
  - Can be used for online learning.

- **Disadvantages:**
  - Not stable: the cost function fluctuates heavily, noisier gradients due to using single samples. It can cause the model to never settle down close to the minimum.
  - Requires a large number of iterations to converge.

### **2.4 Mini-batch Gradient Descent**

- `Mini-batch Gradient Descent (MGD)` splits the training datasets into small batches that are used to calculate model error and update model coefficients.

- Instead of using the whole dataset or a single sample, MGD uses a small number of samples. The number of samples is called the `batch size`. Batch size is a positive integer.
  - It calculates the average gradient of the cost function for the mini-batch and updates the parameters in the opposite direction.
  - If the batch size is equal to the number of samples in the training dataset, mini-batch gradient descent becomes BGD.
  - If the batch size is equal to 1, mini-batch gradient descent becomes SGD.

- MGD is in the middle of BGD and SGD. As a result, it is faster than BGD and more stable than SGD.

- Mini-batch gradient descent follow the steps below:
  - `Step 1`: Randomly initialize the parameters ($`\theta`$)
  - `Step 2`: Randomly select a batch of samples from the training dataset.
    - A mini batch consists of m training samples, where m is between 1 and the full batch size.
    - Common values for m are between 16 and 512.
  - `Step 3`: Calculate the gradient of the cost function based on the selected batch.
    - The loss and gradients are calculated for the entire mini batch of m samples, not just one sample.
    - The gradients are averaged over the m samples to reduce noise.
  - `Step 4`: Move in the direction of the negative gradient to minimize the cost function. Update the parameters ($`\theta`$) based on the gradient of the cost function.
    - $`\theta_{i+1} = \theta_i - \eta \nabla J(\theta_i; \mathbf{x}_{i:i+n}; \mathbf{y}_{i:i+n})`$
  - Step 5: Repeat the process until the result reaches an aceptable value or the maximum number of iterations is reached.
    - After processing one mini batch, a new mini batch is selected.
    - All mini batches are selected in a random shuffle order.

- An example of update equation using MGD for Linear Regression is:

  ```math
  \theta_{j+1} = \theta_j - \eta \frac{1}{b}\sum_{i=1}^{b}(\hat{y}_i - y_i)x^{(i)}_j
  ```

- **Advantages:**
  - Faster than batch gradient descent.
  - Less noisy gradients and More stable than SGD: Averaging gradients over a mini batch reduces noise compared to SGD, which uses a single sample. This leads to smoother optimization.
  - Faster than SGD: Mini batch gradient descent tends to converge faster than SGD since it uses multiple samples to calculate gradients in each step.
  - Balances stability and speed: Mini batch gradient descent finds a good balance between the stability of batch gradient descent and the faster training of stochastic gradient descent.

- **Disadvantages:**
  - Choosing batch size: choosing the right mini batch size requires experimentation.
  - Still need more memory usage than SGD.

### **2.5 Comparison of Gradient Descent algorithms**

![Comparison of Gradient Descent algorithms](https://miro.medium.com/v2/resize:fit:640/format:webp/1*tRhocv_8nr4CwGbc3CaPXQ.jpeg)

| Algorithm | How it Works | Advantages | Disadvantages|
|:--|:--|:--|:--|
| Batch Gradient Descent | Calculates loss and gradients for the entire batch, then does a single parameter update| Fastest convergence; Stable optimization; Gives true gradients| Slow updates; Requires storing full batch in memory|
|Stochastic Gradient Descent | Calculates loss and gradients for 1 example, then immediately does a parameter update. Repeats for each example| Fast updates; Computationally efficient| Noisy gradients; Slow convergence; Unstable optimization|
|Mini-Batch Gradient Descent| Calculates loss and gradients for a mini batch of examples. Averages the gradients, then does a parameter update. Repeats for each mini batch| Faster than batch GD; More stable than SGD; Reduced memory usage| Hard to choose optimal batch size; Less robust than SGD; Still requires more memory|
