# **Densely Connected Convolutional Networks (DenseNets)**

## **1. Introduction**

- DenseNet was proposed in the paper "Densely Connected Convolutional Networks" in 2016.

- DenseNet is a convolutional neural network architecture where each layer is directly connected to every other layer in a feed-forward fashion (within each dense block).

## **2. Architecture**

### **2.1. From ResNets to DenseNets**

- ResNet and DenseNet are both based on the idea of **residual learning**. The difference is that in ResNet, the input of a layer is the sum of the outputs of all previous layers, while in DenseNet, the input of a layer is the concatenation of the outputs of all previous layers.

![ResNet vs DenseNet](https://d2l.ai/_images/densenet-block.svg)

- The key difference between ResNet and DenseNet is that in the latter case outputs are concatenated (denoted by `[,]`) rather than added.

![DenseNet](https://d2l.ai/_images/densenet.svg)

![DenseNet](https://production-media.paperswithcode.com/models/densenet121_spXhNmT.png)

### **2.2. Dense Block**

![Dense Block](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-20_at_11.35.53_PM_KroVKVL.png)

- DenseNets are divided into *Dense Blocks*, in which the dimensions of the feature maps remains constant within a block, but the number of filters changes between them.

- Each dense block consists of multiple **dense layers**. Each dense layer consists of a batch normalization layer, a ReLU activation layer, and a convolutional layer.

- The output of each dense layer is concatenated with the outputs of all previous dense layers and fed into the next dense layer. The output if the last dense layer of a block is the input of the next transition layer.

- **Growth Rate**: since we are concatenating feature maps, this channel dimension is increasing at every layer. The growth rate hyperparameter determines how many filters to add at each layer, despite the increasing number of input feature maps.

- If we make H_l to produce k feature maps every time, then we can generalize for the l-th layer:

```math
k_l = k_0 + k \times (l-1)
```

### **2.3. Transition Layer**

- Since each dense block will increase the number of channels, adding too many of them will lead to an excessively complex model. A transition layer is used to control the complexity of the model.

- It reduces the number of channels by using an 1 x 1 convolution (usually halving the number of channels). Moreover, it halves the height and width via average pooling with a stride of 2.

![Transition Layer](https://www.researchgate.net/publication/349928917/figure/fig3/AS:999690631925761@1615356228754/DenseNet121-transfer-network-Convolution-and-transition-layer-structures.png)

### **2.4. DenseNet models**

- DenseNet model consists of 3 parts:
  - The initial convolutional layer with 64 filters of size 7 x 7 and stride 2. This is followed by a max pooling layer with a pool size of 3 x 3 and stride of 2.
  - The dense block consists of several dense layers. Each dense layer has a growth rate of 32.
  - The transition layer consists of a batch normalization layer, a ReLU activation layer, a 1 x 1 convolutional layer, and a 2 x 2 average pooling layer.
  - The final layer is a 7 x 7 global average pooling layer and a fully connected layer with 1000 outputs.

![DenseNet models](https://iq.opengenus.org/content/images/2021/08/densenet_archs.png)

![DenseNet 121](https://i.imgur.com/4mtAHdI.png)

- In the above figure is the description of DenseNet 121, Dx is dense block, DLx is dense layer, Tx is transition layer, and growth rate is k = 32.

## **3. Implementation**

- This is the implementation of DenseNet 121 in TensorFlow 2.0:

```python
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import (
    AvgPool2D,
    GlobalAveragePooling2D,
    MaxPool2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, Concatenate


class DenseNet:

    densenet_configs = (6, 12, 24, 16)

    def __init__(self, num_classes=1000, input_shape=(224, 224, 3), growth_rate=32, configs = densenet_configs, name='DenseNet121'):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.growth_rate = growth_rate
        self.configs = configs
        self.name = name

    def _base_block(self, x, filters, kernel=1, strides=1):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel, strides=strides, padding="same")(x)
        return x

    def _dense_block(self, x, num_dense_layers, growth_rate):
        for _ in range(num_dense_layers):
            y = self._base_block(x, filters=4*growth_rate, kernel=1, strides=1)
            y = self._base_block(y, filters=growth_rate, kernel=3, strides=1)
            x = Concatenate()([x, y])
        return x

    def _transition_layer(self, x):
        x = self._base_block(x, filters=x.shape[-1]//2, kernel=1, strides=1)
        x = AvgPool2D(2, strides=2, padding="same")(x)
        return x

    def build(self):

        input_layer = Input(shape=self.input_shape)
        x = Conv2D(64, 7, strides=2, padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool2D(3, strides=2, padding="same")(x)

        for num_dense_layers in self.configs:
            d = self._dense_block(x, num_dense_layers = num_dense_layers,  growth_rate = self.growth_rate)
            x = self._transition_layer(d)

        x = GlobalAveragePooling2D()(d)
        output = Dense(self.num_classes, activation="softmax")(x)
        model = Model(inputs=input_layer, outputs=output, name=self.name)
        return model

configs = (6, 12, 24, 16)
densenet121 = DenseNet(num_classes=2, input_shape=(224, 224, 3), growth_rate=32, configs=configs).build()
densenet121.summary()
```
