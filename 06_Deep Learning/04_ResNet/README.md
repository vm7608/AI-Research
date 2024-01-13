# **Residual Networks (ResNets)**

## **1. Introduction**

- ResNet (Residual Networks) was first introduced in 2015 in the paper – “Deep Residual Learning for Image Recognition”.

- A ResNet can be called an upgraded version of the VGG architecture, with the difference between them being the skip connections used in ResNets.

## **2. Architecture**

### **2.1. Problems**

- Deep Neural Networks provide more accuracy as the number of layers increases. But, when we go deeper into the network, the accuracy of the network decreases instead of increasing. This is known as the `degradation problem`.

- The main reason behind this is the `vanishing gradient problem`. As we go deeper into the network, the gradient keeps on decreasing until it becomes almost zero. This makes the training process very slow and the network is not able to learn efficiently.

- This was one of the bottlenecks of VGG and previous models. They couldn’t go as deep as wanted, because they started to lose generalization capability.

![Deeper model cause error](https://iq.opengenus.org/content/images/2020/03/Screenshot-from-2020-03-20-13-21-33.png)

### **2.2. Function Classes**

- Consider:
  - $`\mathcal{F}`$ is the functions classes that a specific network architecture can reach. That is $`\forall f \in \mathcal{F}`$, there exists some set of parameters that can be obtained through training on a suitable dataset
  - $`f^*`$ is the "truth" function that we really would like to find.
  - We will try to find some $`f^*_\mathcal{F}`$ which is our best within $`\mathcal{F}`$.

- If we design a different and more powerful architecture $`\mathcal{F}'`$, we would expect that $`f^*_{\mathcal{F}'}`$ is "better" than $`f^*_{\mathcal{F}}`$. However, if $`\mathcal{F} \not\subseteq \mathcal{F}'`$ there is no guarantee that this should even happen. In fact, $`f^*_{\mathcal{F}'}`$ might well be worse.

- Consider the following firgure:
  - For non-nested function classes, a larger function class does not always move closer to the "truth" function $`f^*`$.
  - For nested function classes where $`\mathcal{F}_1 \subseteq \ldots \subseteq \mathcal{F}_6`$ on the right, we can avoid the above issue from the non-nested function classes.

![Function classes](https://d2l.ai/_images/functionclasses.svg)

### **2.5. Residual Block**

![Residual block](https://d2l.ai/_images/residual-block.svg)

- Let's consider the above figure:
  - $`\mathbf{x}`$ is the input to the block.
  - Assume that the desired underlying mapping we want to obtain by learning is $`f(\mathbf{x})`$.
  - On the left, the block must directly learn the desired underlying mapping $`f(\mathbf{x})`$.

  - On the right, the block needs to learn the `residual mapping` $`g(\mathbf{x}) = f(\mathbf{x}) - \mathbf{x}`$.

- The right figure illustrates the *residual block* of ResNet, where the solid line carrying the layer input $\mathbf{x}$ to the addition operator is called a *residual connection* (or *shortcut connection*).

- This help the networks only need to learn the `residual mapping` $`(g(\mathbf{x}) = f(\mathbf{x}) - \mathbf{x})`$ instead of the desired underlying mapping $`f(\mathbf{x})`$. In other words, this allows the network to learn the difference between the input and the desired output instead of trying to approximate the output directly.

- The intuition is that if the identity mapping is optimal ($`f(\mathbf{x}) = \mathbf{x}`$ is the function we want to learn), it is easier to push the residual to zero $`(g(\mathbf{x}) = 0)`$) than to fit an identity mapping by a stack of nonlinear layers (like on the left).

- ResNet follows VGG’s full convolutional layer design:
  - A residual block has some convolutional layers followed by a batch normalization layer and a ReLU activation function.
  - Then, we skip these two convolution operations and add the input directly before the final ReLU activation function.

#### **2.5.1. Identity Block**

- The identity block is used when the input activation has the same dimension as the output activation.

- Example of identity block in ResNet 34 (left)

![Identity block](https://d2l.ai/_images/resnet-block.svg)

#### **2.5.2. Bottleneck / Convolutional block**

- When the input and output dimensions don’t match up, we can use a convolutional block.

- The convolutional layer in the shortcut path is used to resize the input $`\mathbf{x}`$ to a different dimension, so that the dimensions match up for the addition operation. We can use a $`1 \times 1`$ convolutional layer with a stride of $`s`$ to change the dimension.

- The $`1 \times 1`$ convolutional layer is also called a `projection layer` because it is used to change the dimension of the input $`\mathbf{x}`$.

- Example of bottleneck block in ResNet 34 (right)

![Convolutional block](https://d2l.ai/_images/resnet-block.svg)

### **2.6. ResNet models**

- There are 3 main components that make up the ResNet. They are:
  - The first convolutional layer (conv1) include a $`7 \times 7`$ convolutional layer with 64 filters, followed by a $`3 \times 3`$ max pooling layer with a stride of 2.
  - The second component is the stack of residual blocks (conv2 ~ conv5).
  - The third component is the global average pooling layer and the fully connected layer.

![ResNet models](https://img2018.cnblogs.com/blog/1203819/201905/1203819-20190528200653959-787225727.jpg)

- The number of parameters in ResNet 18/34/50/101/152 are 11.2M/21.3M/23.5M/42.7M/58.3M respectively.

- ResNet use stride of 2 to reduce the height and width of the output instead of a pooling operation. Only one max pooling operation is performed in our Conv1 layer, and one average pooling layer at the end of the ResNet. Global average pooling calculates the mean of each feature map and then concatenates all the values into a one-dimension list.

![Global average pooling](https://static.hindawi.com/articles/sv/volume-2021/1205473/figures/1205473.fig.002.jpg)

- There was a small difference between ResNet 50/101/152 and ResNet 18/34 is that before the shortcut connections skipped two layers but now they skip three layers and also there was 1 * 1 convolution layers added with the purpose of reducing the dimensionality of the input to the next layer.

![ResNet 18/34 block vs ResNet 50](https://fastai.github.io/fastbook2e/images/att_00045.png)

## **3. ResNetXt**

- One of the challenges in the design of ResNet is the trade-off between nonlinearity and dimensionality within a given block:
  - Having enough nonlinearity within a block to learn useful transformations

  - Keeping the dimensionality and number of parameters within a reasonable limit

  - Not making the blocks too deep to hamper optimization

- To solve this problem, a possible solution is increase the number of channels that can carry information between blocks

- This solution have one problem is that it increases the number of parameters and the computational cost of the network. So, ResNeXt solves this problem by using a `split-transform-merge` strategy.

![ResNeXt](https://d2l.ai/_images/resnext-block.svg)

- The solution of ResNeXt is to replace the convolutional layers with a set of parallel branches, each of which transforms its input by a separate sequence of transformations. The outputs of the branches are aggregated by summation or concatenation and then fed into the next layer.

## **4. Implement**

- The following code implements ResNet-50 from scratch.
  - The identity block:
  
  ![Indentity block](https://i.stack.imgur.com/37qzA.png)
  
  - The convolutional block:

  ![Convolutional block](https://i.stack.imgur.com/0mE2p.png)

```python
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D
from tensorflow.keras.layers import Add, ReLU, Dense, Dropout
from tensorflow.keras import Model


class ResNet:

    resnet_configs = ((3, 64), (4, 128), (6, 256), (3, 512))
    
    def __init__(self, num_classes=1000, input_shape=(224, 224, 3), configs = resnet_configs, name='ResNet50'):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.configs = configs
        self.name = name

    def _conv_batchnorm_relu(self, x, filters, kernel_size, strides=1):
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def _identity_block(self, tensor, filters):
        x = self._conv_batchnorm_relu(
            tensor, filters=filters, kernel_size=1, strides=1)

        x = self._conv_batchnorm_relu(
            x, filters=filters, kernel_size=3, strides=1)

        x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)
        x = BatchNormalization()(x)

        x = Add()([tensor, x])  # skip connection
        x = ReLU()(x)
        return x

    def _convolutional_block(self, tensor, filters, strides):
        # left stream
        x = self._conv_batchnorm_relu(
            tensor, filters=filters, kernel_size=1, strides=strides)

        x = self._conv_batchnorm_relu(
            x, filters=filters, kernel_size=3, strides=1)

        x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)
        x = BatchNormalization()(x)

        # right stream
        shortcut = Conv2D(filters=4*filters, kernel_size=1,
                          strides=strides)(tensor)
        shortcut = BatchNormalization()(shortcut)

        x = Add()([shortcut, x])  # skip connection
        x = ReLU()(x)
        return x

    def _resnet_block(self, x, filters, reps, strides):
        x = self._convolutional_block(x, filters, strides)
        for _ in range(reps-1):
            x = self._identity_block(x, filters)
        return x

    def build(self):

        # ResNet50
        input_layer = Input(shape=self.input_shape)

        x = self._conv_batchnorm_relu(input_layer, filters=self.configs[0][1], kernel_size=7, strides=2)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        for i in range(len(self.configs)):
            x = self._resnet_block(
                x, filters= self.configs[i][1], reps=self.configs[i][0], strides=1 if i == 0 else 2)

        x = GlobalAvgPool2D()(x)
        output = Dense(units=self.num_classes, activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output, name=self.name)

        return model


configs = ((3, 64), (4, 128), (6, 256), (3, 512))
resnet50 = ResNet(num_classes=2, input_shape=(224, 224, 3), configs=configs).build()
resnet50.count_params()
```
