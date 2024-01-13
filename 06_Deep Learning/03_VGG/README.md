# **VGG Network Architecture**

## **1. Introduction**

- The VGG network is a deep convolutional neural network that was developed by researchers at the `Visual Geometry Group (VGG)` at the University of Oxford in 2014.

- While AlexNet offered empirical evidence that `deep CNNs can achieve good results`, it did not provide a general template to guide subsequent researchers in designing new networks.

- So VGGNet was designed to provide a general template when designing a CNN architecture:

  - Providing a general architecture for deep CNNs.
  - Reduce the number of parameters in the CONV layers and improve on training time.
  - Increase the depth of the network.
  - Consistency and simplicity in the architecture by using `blocks`, which are a sequence of convolutional layers with 3x3 kernels and a max-pooling layer with 2x2 kernels.

- One of the `key contributions` of the VGG network is its use of _small 3x3 filters_ in the convolutional layers, which increases the depth of the network while keeping the number of parameters relatively low.

## **2. Architecture**

### **2.1. VGG Blocks**

- The basic building block of CNNs is a sequence of the following:

  - (i) convolutional layers with padding to maintain the resolution.
  - (ii) a nonlinearity such as a ReLU
  - (iii) a pooling layer such as max-pooling to reduce the resolution.

- The key idea of Simonyan and Zisserman (2014) was to use multiple convolutions in between downsampling via max-pooling in the form of a block. The successive application of two 3x3 convolutions touchs the same receptive field as a single 5x5 convolution but with fewer parameters.

  - First - The benefit of using multiple smaller layers is that more non-linear activation layers accompany the convolution layers, improving the decision functions and allowing the network to converge quickly.
  - Second, VGG uses a smaller convolutional filter, which reduces the network’s tendency to over-fit during training.
  - Third, A 3×3 filter is the optimal size (the smallest possible receptive field) that can capture left-right and up-down information. A 5×5 filter would capture more information, but it would be more expensive. Consistent 3×3 convolutions make the network easy to manage.

- A VGG block consists of:
  - A sequence of convolutions with 3 x 3 kernels with padding of 1 (keeping height and width)
  - Followed by a 2 x 2 max-pooling layer with stride of 2 (halving height and width after each block).

![VGG Block](https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcSIkAkIEk8jBYfGhutkDx3qfkBMjtqBqRvEzTkhGldgetVJix5_)

#### Here is a quick outline of the VGG architecture

- Input—VGGNet receives a 224×224 image input.

- Convolutional layers:

  - Kernel size: 3×3
  - Stride: 1
  - Padding: same
  - Number of filters: 64, 128, 256, 512, 512

- Activation function: VGGNet uses ReLU for the activation function. This is a significant improvement over the sigmoid and tanh activation functions used in previous networks. ReLU is `faster to compute` and does not suffer from the `vanishing gradient problem`.

- Pooling layer:

  - Kernel size: 2×2
  - Stride: 2
  - Max pooling
  - Padding: same

- Fully connected layers:
  - Include 3 fully connected layers.
  - The first two fully connected layers each have 4096 channels.
  - The third fully connected layer has 1000 channels, one for each class. (The ImageNet dataset has 1000 classes.)

### **2.2. VGG Network**

- The VGG Network can be partitioned into two parts:
  - The first consisting mostly of convolutional and pooling layers designed in blocks
  - The second consisting of fully connected layers

#### **VGG-16**

- VGG16 is a 16-layer deep neural network. It contains total of 138M parameters but in a simplicity of architecture.

![VGG16 Architecture](https://datagen.tech/wp-content/uploads/2022/11/image2-1.png)

![VGG16 Architecture](https://raw.githubusercontent.com/blurred-machine/Data-Science/master/Deep%20Learning%20SOTA/img/config2.jpg)

#### **VGG-19**

- The only difference between VGG16 and VGG19 is that VGG19 has 3 additional convolutional layers, all of which are 3x3.
- The additional layers are incorporated into the middle of the network, these layers do not change the input and output sizes of the convolutional layers, but they do increase the number of parameters in the network.

![VGG19 Architecture](https://miro.medium.com/max/1658/1*4F-9zrU07yhwj6gChX_q-Q.png)

#### **All VGG configurations**

![All VGG configurations](https://debuggercafe.com/wp-content/uploads/2021/08/different-vgg-models..png)

![All VGG configurations](https://th.bing.com/th/id/OIP.OQPqpH0vFTPvxldqLY1oxwHaBJ?pid=ImgDet&rs=1)

## **3. Implement**

- This is the implementation of VGG16 using Tensorflow 2.0

```python
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, ReLU
from tensorflow.keras import Model
class VGGNet:

    vgg16_configs = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))

    def __init__(self, num_classes=1000, input_shape=(224, 224, 3), configs = vgg16_configs, name='VGG16'):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.configs = configs
        self.name = name

    def _vgg_block(self, x, filters, num_convs):
        for _ in range(num_convs):
            x = Conv2D(filters=filters, kernel_size =3, padding ='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
        x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
        return x

    def build(self):

        # VGGNet
        input_layer = Input(shape=self.input_shape)
        x = input_layer
        for (num_conv, filter) in self.configs:
            x = self._vgg_block(x, filters=filter, num_convs=num_conv)

        # Fully connected layers
        x = Flatten()(x)
        x = Dense(units=256, activation='relu')(x)
        x = Dense(units=256, activation='relu')(x)
        output = Dense(units=self.num_classes, activation='softmax')(x)

        # Create the model
        model = Model(inputs=input_layer, outputs=output, name=self.name)
        return model

configs = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
vgg16 = VGGNet(num_classes=2, input_shape=(224, 224, 3), configs=configs).build()
vgg16.summary()
```
