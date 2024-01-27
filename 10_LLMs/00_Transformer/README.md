# **Transformer Architecture**

## **1. Introduction and Overview**

### **1.1. Limitations of previous models**

#### **1.1.1. RNN**

RNN stands for Recurrent Neural Network. RNNs are a type of artificial neural network that is good at handling sequential data like text, time series, video, audio etc.

The key aspect of RNNs is that they have a hidden state or memory that remembers information about what has been calculated so far. This allows them to recognize patterns in sequence data, such as text, speech, time series, etc.

RNNs process one input at a time and update their hidden state after seeing each input. The hidden state captures information about the sequence seen so far. This hidden state is then used along with the next input to calculate the next hidden state, and so on for the entire sequence. This makes RNNs great at multiple types of tasks:

1. Vector-Sequence Models — Input a fixed-sized vectors and output vectors of any size. For example, in image captioning, the image is the input and the output describes the image.
2. Sequence-Vector Model — Input a vector of any size and output a vector of fixed size. For example, sentiment analysis of a movie rates the review of any movie is fixed at positive or negative.
3. Sequence-to-Sequence Model — Input a sequence and outputs another sequence with variant sizes. An example of this is language translation.

<p align="center">
  <img src="https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/Transformer-neural-network-2.png" >
  <br>
  <i>Architecture of RNNs</i>
</p>

However RNN has two major disadvantages

1. It’s slow to train.
2. Long sequences lead to vanishing gradient or the problem of long-term dependencies. In simple terms, its memory is not that strong when it comes to remembering old connections.

For example, in the sentence  “The clouds are in the __.” the next word should obviously be "sky", as it is linked with the clouds. If the distance between clouds and the predicted word is short, so the RNN can predict it easily.

Consider another example: “I grew up in Germany with my parents, I spent many years there and have proper knowledge about their culture. That’s why I speak fluent __.”. Here the predicted word is "German", but distance between Germany and the predicted word is longer in this case, so it’s difficult for the RNN to predict.

So, unfortunately, as that gap grows, RNNs become unable to connect as their memory fades with distance.

<p align="center">
  <img src="https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/Transformer-neural-network-3.png" >
  <br>
  <i>Image illustrating long-term dependencies.</i>
</p>

#### **1.1.2. LSTM**

Long short-term memory is a special kind of RNN, specially made for solving vanishing gradient problems. They are capable of learning long-term dependencies.

<p align="center">
  <img src="https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/Transformer-neural-network-4.png" >
  <br>
  <i>Architecture of LSTM</i>
</p>

LSTM neurons have a branch that allows passing information to skip the long processing of the current cell. This branch allows the network to retain memory for a longer period of time. It improves the vanishing gradient problem but not terribly well: It will do fine until 100 words, but around 1,000 words, it starts to lose its memory.

Further, like the simple RNN, it is also very slow to train, and perhaps even slower than RNN. These systems take input sequentially one by one, which doesn’t use up GPUs very well, which are designed for parallel computation. Later, I’ll address how we can parallelize sequential data.

To sum up, LSTM faces 2 major problems like RNN:

- Vanishing gradient
- Slow training

### **1.2. Transformer solves the problems**

The transformer neural network is a novel architecture that aims to solve sequence-to-sequence tasks while handling long-range dependencies with ease. It was first proposed in the paper “Attention Is All You Need.”  and became a state-of-the-art technique in the field of NLP.

Main advantages of the transformer architecture are:

- The input sequence can be passed parallelly so that GPU can be used effectively and the speed of training can also be increased.
- Transformer based on the multi-headed attention technique, so it easily overcomes the vanishing gradient issue.

The following table is a comparison of the three models discussed above:

| **Aspect** | **RNN** | **LSTM** | **Transformer** |
|-----------|-----------|-----------|-----------|
| Input Processing | Sequential processing | Sequential processing     | Parallel processing             |
| Long-Term Dependencies | Struggles with long-range dependencies | Improved at capturing long-range dependencies | Efficiently captures long-range dependencies |
| Parameter Efficiency | Requires a moderate number of parameters | Requires a moderate number of parameters | Competitive performance with fewer parameters |
| Limited Context | Limited context window | Limited context window | Provides global context |
| Efficiency and Scalability | Can be computationally expensive | Can be computationally expensive | More computationally efficient and scalable |
| Positional Information | Does not inherently encode positional information | Does not inherently encode positional information | Explicitly encodes positional information |
| Interpretable Representations | Less interpretable representations | Less interpretable representations | More interpretable representations due to self-attention |

## **2. Transformer Architecture**

### **2.1. High-level Overview**

The Transformer architecture uses an `encoder-decoder structure` that does not rely on recurrence and convolutions to generate an output. The encoder maps an input sequence to a series of continuous representations. The decoder receives the encoder’s output and the decoder’s output at a previous time step, and generates an output sequence.

<p align="center">
  <img src="https://datagen.tech/wp-content/uploads/2023/01/image1.png" >
  <img src="http://jalammar.github.io/images/t/The_transformer_encoders_decoders.png" >
  <br>
  <i>Transformer Architecture</i>
</p>

The encoding component is a stack of encoders (The original research paper used 6 Encoders and 6 Decoders). The decoding component is a stack of decoders of the same number. The condition is — The number of Encoders and number of Decoders should be same.

<p align="center">
  <img src="http://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png" >
  <br>
  <i>The original model consists of 6-Encoders and 6-Decoders</i>
</p>

### **2.2. Encoder**

Each ENCODER network consists of 2 parts — Self-attention and feed-forward-neural-networks.

<p align="center">
  <img src="http://jalammar.github.io/images/t/Transformer_encoder.png" >
  <br>
  <i>Encoder structure</i>
</p>

The encoder’s inputs first flow through a self-attention layer – a layer that helps the encoder look at other words in the input sentence as it encodes a specific word. The outputs of the self-attention layer are fed to a feed-forward neural network. The exact same feed-forward network is independently applied to each position. Feed forward neural network is a fully connected MLP (multi-layered perceptron), which consists of 512 hidden units and a ReLU activation function.

As is the case in NLP applications in general, we begin by turning each input word into a vector using an embedding algorithm.

<p align="center">
  <img src="http://jalammar.github.io/images/t/embeddings.png" >
  <br>
  <i>Each word is embedded into a vector of size 512.</i>
</p>

The embedding only happens in the bottom-most encoder. The abstraction that is common to all the encoders is that they receive a list of vectors each of the size 512. In the bottom encoder that would be the word embeddings, but in other encoders, it would be the output of the encoder that’s directly below. The size of this list is hyperparameter we can set – basically it would be the length of the longest sentence in our training dataset.

After embedding the words in input sequence, each of word flows through each of the two layers of the encoder. Here we begin to see one key property of the Transformer, which is that the word in each position flows through its own path in the encoder. The only connection between these paths is in the self-attention layer.

All these 512-dimensional encoded vectors are passed as an input to the first Encoder unit. The output of this Encoder unit will be passed as an input to the next Encoder unit.

All the inputs are passed at the same time to the self-attention layer in the encoder network. It is not like recurrent networks, where the inputs are provided at each timesteps so it makes the training process faster.

<p align="center">
  <img src="http://jalammar.github.io/images/t/encoder_with_tensors_2.png" >
  <br>
  <i>The word at each position passes through a self-attention process. Then, they each pass through a feed-forward neural network -- the exact same network with each vector flowing through it separately.</i>
</p>

### **2.3. Attention in Transformer (Self-Attention)**

#### **2.3.1. Self-Attention at a High Level**

Say the following sentence is an input sentence we want to process:

> "The animal didn't cross the street because it was too tired”

What does "it" in this sentence refer to? Is it referring to the street or to the animal? It’s a simple question to a human, but not as simple to an algorithm.

When the model is processing the word "it", self-attention will associate “it” with “animal”. As the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding/understanding for this word.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*pT0ZIWeoilLkz3e_1fVeYQ.png" >
  <br>
  <i>Self-attention allows the model to look at other words in the input sentence as it encodes a specific word. Here darker colors represent higher attention </i>
</p>

As discussed above, our task is to generate $`z_i`$ given $`x_i`$ — to do this task, the attention should be `on the other words in the same sentence`. For instance: In the input sentence — The animal didn’t cross the street because it was too tired — the word “it” in this sentence is $`x_i`$ and $`z_i`$ should be generated based on this $`x_i`$. To generate $`z_i`$ — the attention should be on the remaining words in the same sentence. This is called self-attention.

#### **2.3.1. Scaled Dot-Product Attention in Detail**

`The first step` in calculating self-attention is to create 3 vectors from each of the encoder’s input vectors. So for each word, we create a `Query vector`, a `Key vector`, and a `Value vector`. These vectors are created by multiplying the embedding by three matrices that we trained during the training process.

Notice that these new vectors are smaller in dimension than the embedding vector. Their dimensionality is 64, while the embedding and encoder input/output vectors have dimensionality of 512.

<p align="center">
  <img src="http://jalammar.github.io/images/t/transformer_self_attention_vectors.png" >
  <br>
  <i>Multiplying x1 by the WQ weight matrix produces q1, the "query" vector associated with that word. We end up creating a "query", a "key", and a "value" projection of each word in the input sentence.</i>
</p>

`The second step` in calculating self-attention is to calculate a score. Say we’re calculating the self-attention for the first word in this example, “Thinking”. We need to score each word of the input sentence against this word. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position.

The score is calculated by taking the dot product of the query vector with the key vector of the respective word we’re scoring. So if we’re processing the self-attention for the word in position #1, the first score would be the dot product of q1 and k1. The second score would be the dot product of q1 and k2.

<p align="center">
  <img src="http://jalammar.github.io/images/t/transformer_self_attention_score.png" >
  <br>
  <i>Calculating the self-attention score for the word.</i>
</p>

`The third and fourth steps` are to divide the scores by 8 (the square root of the dimension of the key vectors used in the paper – 64. This leads to having more stable gradients. There could be other possible values here, but this is the default), then pass the result through a softmax operation. Softmax normalizes the scores so they’re all positive and add up to 1.

<p align="center">
  <img src="http://jalammar.github.io/images/t/self-attention_softmax.png" >
  <br>
  <i>Dividing the scores by 8 (to stabilize gradients), then passing them through a softmax operation.</i>
</p>

This softmax score determines how much each word will be expressed at this position. Clearly the word at this position will have the highest softmax score, but sometimes it’s useful to attend to another word that is relevant to the current word.

`The fifth step` is to multiply each value vector by the softmax score (in preparation to sum them up). The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny numbers like 0.001, for example).

`The sixth step` is to sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word).

<p align="center">
  <img src="http://jalammar.github.io/images/t/self-attention-output.png" >
  <br>
  <i>Multiplying each value vector by the softmax score (to focus on the important words), then summing them up (to create the output).</i>
</p>

That concludes the self-attention calculation. The resulting vector is then sent along to the feed-forward neural network. In the actual implementation, however, this calculation is done in matrix form for faster processing. We’ll go over that next.

#### **2.3.2. Matrix Calculation of Self-Attention**

`The first step` is to calculate the Query, Key, and Value matrices. We do that by packing our embeddings into a matrix X with each row is an embedding vector of a word in input sentence, and multiplying it by the weight matrices we’ve trained (WQ, WK, WV).

<p align="center">
  <img src="http://jalammar.github.io/images/t/self-attention-matrix-calculation.png" >
  <br>
  <i>Every row in the X matrix corresponds to a word in the input sentence. We again see the difference in size of the embedding vector (512, or 4 boxes in the figure), and the q/k/v vectors (64, or 3 boxes in the figure)</i>
</p>

Finally, since we’re dealing with matrices, we can condense steps two through six in one formula to calculate the outputs of the self-attention layer.

<p align="center">
  <img src="http://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png" >
  <br>
  <i>The self-attention calculation in matrix form</i>
</p>

#### **Additional explain for Key, Query and Value in Self-Attention**

Self-attention is processed along the path of each token in the segment. The significant components are three vectors:

- Query: The query is a representation of the current word used to score against all the other words (using their keys). We only care about the query of the token we’re currently processing.
- Key: Key vectors are like labels for all the words in the segment. They’re what we match against in our search for relevant words.
- Value: Value vectors are actual word representations, once we’ve scored how relevant each word is, these are the values we add up to represent the current word.

An example is to think of it like searching through a filing cabinet. The query is like a sticky note with the topic you’re researching. The keys are like the labels of the folders inside the cabinet. When you match the tag with a sticky note, we take out the contents of that folder, these contents are the value vector. Except you’re not only looking for one value, but a blend of values from a blend of folders. Multiplying the query vector by each key vector produces a score for each folder (technically: dot product followed by softmax).

<p align="center">
  <img src="http://jalammar.github.io/images/gpt2/self-attention-example-folders-3.png" >
  <img src="http://jalammar.github.io/images/gpt2/self-attention-example-folders-scores-3.png" >
  <br>
  <i>Example of Query/Key/value</i>
</p>

We multiply each value by its score and sum up – resulting in our self-attention outcome. This weighted blend of value vectors results in a vector that paid 50% of its “attention” to the word robot, 30% to the word a, and 19% to the word it.

<p align="center">
  <img src="http://jalammar.github.io/images/gpt2/gpt2-value-vector-sum.png" >
  <br>
  <i>Calculate scores</i>
</p>

#### **2.3.3. Multi-Head Attention**

The paper further develop the self-attention layer by adding a mechanism called `“multi-headed” attention`. This improves the performance of the attention layer in two ways:

1. `It expands the model’s ability to focus on different positions`. In the example above, $`z_1`$ contains a little bit of every other encoding vector, but it could be dominated by the actual word itself. (This means z1 theoretically contains information from all other words, but in practice it is often just reflecting the encoding of the input word itself rather than learning meaningful relationships between different words.). Multi-headed attention helps address this by allowing the model to learn attending to different parts of the sequence for each attention head. So some heads can focus just on the current word, while others can learn long-range dependencies.

2. `It gives the attention layer multiple “representation subspaces”`. With multi-headed attention we have not only one, but multiple sets of Query/Key/Value weight matrices (the Transformer uses eight attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.

<p align="center">
  <img src="http://jalammar.github.io/images/t/transformer_attention_heads_qkv.png" >
  <br>
  <i>With multi-headed attention, we maintain separate Q/K/V weight matrices for each head resulting in different Q/K/V matrices. As we did before, we multiply X by the WQ/WK/WV matrices to produce Q/K/V matrices.</i>
</p>

If we do the same self-attention calculation we outlined above, just eight different times with different weight matrices, we end up with eight different Z matrices. This leaves us with a bit of a challenge. The feed-forward layer is not expecting eight matrices – it’s expecting a single matrix (a vector for each word). So we need a way to condense these eight down into a single matrix.

<p align="center">
  <img src="http://jalammar.github.io/images/t/transformer_attention_heads_z.png" >
  <br>
  <i>Each Z matrix is a different representation of the input.</i>
</p>

To solve that, we concat the matrices then multiply them by an additional weights matrix WO.

<p align="center">
  <img src="http://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png" >
  <br>
  <i>Concatenating the Z matrices, then multiplying them by WO.</i>
</p>

<p align="center">
  <img src="http://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png" >
  <br>
  <i>Putting all the pieces together.</i>
</p>

### **2.4. Positional Encoding**

One thing that’s missing from the model is a way to account for the order of the words in the input sequence.

To address this, the transformer adds a vector to each input embedding. These vectors helps the model determine the position of each word, or the distance between different words in the sequence. The intuition here is that adding these values to the embeddings provides meaningful distances between the embedding vectors once they’re projected into Q/K/V vectors and during dot-product attention.

<p align="center">
  <img src="http://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png" >
  <img src="http://jalammar.github.io/images/t/transformer_positional_encoding_example.png" >
  <br>
  <i>Positional encoding vectors are added to the embedding vector. The final dimension of the vector is 512 (the same as the embedding vector). The positional encoding vector is added to the embedding vector.</i>
</p>

It is important to acknowledge that t1, t2, t3,.. are static vectors with each vector being 512 dimensions. Static vectors means — these vectors are NOT TRAINABLE (there are no trainable parameters). These vectors are designed in a way, such that the distance between t1 vector and t2 vector is less than the distance between t1 vector and t3 vector. Similarly, the distance between t2 vector and t3 vector is less than the distance between t2 vector and t4 vector. The similar pattern and theory is applicable for all the position encoded vectors using the formula shown below.

<p align="center">
  <img src="https://camo.githubusercontent.com/6adc8b88ed006c582e47c21a0b9f860bcc73535b402ed376a8adf48ee0d56f56/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f6d61782f323137342f302a507556383749566b46696e3938455a572e706e67" >
  <img src="http://jalammar.github.io/images/t/attention-is-all-you-need-positional-encoding.png" >
  <br>
  <i>Positional encoding formula</i>
</p>

For every odd index on the input vector, create a vector using the cos function. For every even index, create a vector using the sin function. "d" is the dimension of embedding vector and "i" is the index of the dimension.

### **2.5. Residual Connections**

One detail in the architecture of the encoder is that each sub-layer (self-attention, ffnn) in each encoder has a residual connection around it, and is followed by a layer-normalization step. If we’re to visualize the vectors and the layer-norm operation associated with self attention, it would look like this:

<p align="center">
  <img src="http://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png" >
  <br>
  <i>Residual connections and layer normalization applied to self-attention.</i>
</p>

This goes for the sub-layers of the decoder as well. If we’re to think of a Transformer of 2 stacked encoders and decoders, it would look something like this:

<p align="center">
  <img src="http://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png" >
  <br>
  <i>Example of 2 stacked encoders and decoders</i>
</p>

### **2.6. Feed-Forward Networks**

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

```math
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

Where W_1, b_1, W_2, and b_2 are the weights and biases of the two linear layers. The dimensionality of input and output is d_model = 512, and the inner-layer has dimensionality d_ff = 2048.

### **2.7. Decoder**

#### **2.7.1. High level view of Decoder**

The encoder start by processing the input sequence. The output of the top encoder is then transformed into a set of attention vectors K and V. These are to be used by each decoder in its “encoder-decoder attention” layer which helps the decoder focus on appropriate places in the input sequence.

The following steps repeat the process until a special symbol is reached indicating the transformer decoder has completed its output. The output of each step is fed to the bottom decoder in the next time step, and the decoders bubble up their decoding results just like the encoders did. And just like we did with the encoder inputs, we embed and add positional encoding to those decoder inputs to indicate the position of each word.

<p align="center">
  <img src="http://jalammar.github.io/images/t/transformer_decoding_1.gif" >
  <br>
</p>

<p align="center">
  <img src="http://jalammar.github.io/images/t/transformer_decoding_2.gif" >
  <br>
  <i>After finishing the encoding phase, we begin the decoding phase. Each step in the decoding phase outputs an element from the output sequence (the English translation sentence in this case).</i>
</p>

The self attention layers in the decoder operate in a slightly different way than the one in the encoder:

- In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence. This is done by masking future positions (setting them to -inf) before the softmax step in the self-attention calculation.

- The “Encoder-Decoder Attention” layer works just like multiheaded self-attention, except it creates its Queries matrix from the layer below it, and takes the Keys and Values matrix from the output of the encoder stack.

#### **2.6.1. Decoders First Multi-Headed Attention - Masked Multi-Head Attention**

The beginning of the decoder is pretty much the same as the encoder. The input goes through an embedding layer and positional encoding layer to get positional embeddings. The positional embeddings get fed into the first multi-head attention layer which computes the attention scores for the decoder’s input. But, there is a slight difference in how the attention scores are calculated.

Since the decoder is `autoregressive and generates the sequence word by word`, you need to prevent it from conditioning to future tokens. For example, when computing attention scores on the word “am”, you should not have access to the word “fine”, because that word is a future word that was generated after. The word “am” should only have access to itself and the words before it. This is true for all other words, where they can only attend to previous words.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/0*0pqSkWgSPZYr_Sjx.png" >
  <br>
  <i>A depiction of Decoder’s first Multi-headed Attention scaled attention scores. The word “am”, should not any values for the word “fine”. This is true for all other words.</i>
</p>

We need a method to prevent computing attention scores for future words. This method is called masking. To prevent the decoder from looking at future tokens, you apply a look ahead mask. The mask is added before calculating the softmax, and after scaling the scores.

##### **Look-Ahead Mask**

The mask is a matrix that’s the same size as the attention scores filled with values of 0’s and negative infinities. When you add the mask to the scaled attention scores, you get a matrix of the scores, with the top right triangle filled with negativity infinities.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*QYFua-iIKp5jZLNT.png" >
  <br>
  <i>Adding a look-ahead mask to the scaled scores</i>
</p>

The reason for the mask is because once you take the softmax of the masked scores, the negative infinities get 0 out, leaving 0 attention scores for future tokens. As the figure below, the attention scores for “am”, has values for itself and all words before it but is zero for the word “fine”. This essentially tells the model to put no focus on those words.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*3ykVCJ9okbgB0uUR.png" >
  <br>
  <i>Applying softmax function to the masked values</i>
</p>

This masking is the only difference in how the attention scores are calculated in the first multi-headed attention layer. This layer still has multiple heads, that the mask is being applied to, before getting concatenated and fed through a linear layer for further processing. The output of the first multi-headed attention is a masked output vector with information on how the model should attend on the decoder’s input.

The masking makes the decoder unidirectional (unlike the bidirectional encoder). This means that the decoder can only attend to previous words in the output sequence. This is why the decoder is autoregressive, because it generates the output sequence one word at a time, and can only attend to previous words.

#### **2.6.2. Encoder-Decoder Attention**

The second layer implements a multi-head self-attention mechanism similar to the one implemented in the first sublayer of the encoder. On the decoder side, this multi-head mechanism receives the queries from the previous decoder sublayer and the keys and values from the output of the encoder. This allows the decoder to attend to all the words in the input sequence.

#### **2.6.3. Feed-Forward Networks**

The third layer implements a fully connected feed-forward network, similar to the one implemented in the second sublayer of the encoder.

#### **2.6.4. The Final Linear and Softmax Layer**

The decoder stack outputs a vector of floats. How do we turn that into a word? That’s the job of the final Linear layer which is followed by a Softmax Layer.

The Linear layer is a simple fully connected neural network that projects the vector produced by the stack of decoders, into a much, much larger vector called a logits vector.

Let’s assume that our model knows 10,000 unique English words (model’s “output vocabulary”) that it’s learned from its training dataset. This would make the logits vector 10,000 cells wide – each cell corresponding to the score of a unique word. That is how we interpret the output of the model followed by the Linear layer.

The softmax layer then turns those scores into probabilities (all positive, all add up to 1.0). The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.

<p align="center">
  <img src="http://jalammar.github.io/images/t/transformer_decoder_output_softmax.png" >
  <br>
  <i>The final linear and softmax layer turns the decoder stack output into probabilities for each word in the output vocabulary.</i>
</p>

## **3. Training**

The Transformer’s goal is to learn how to output the target sequence, by using both the input and target sequence. The Transformer processes the data like this:

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*0g4qdq7Rt6QvDalFFAkL5g.png" >
  <br>
  <i>Training the Transformer model</i>
</p>

1. The input sequence is converted into Embeddings (with Position Encoding) and fed to the Encoder.
2. The stack of Encoders processes this and produces an encoded representation of the input sequence.
3. The target sequence is prepended with a start-of-sentence token, converted into Embeddings (with Position Encoding), and fed to the Decoder.
4. The stack of Decoders processes this along with the Encoder stack’s encoded representation to produce an encoded representation of the target sequence.
5. The Output layer converts it into word probabilities and the final output sequence.
6. The Transformer’s Loss function compares this output sequence with the target sequence from the training data. This loss is used to generate gradients to train the Transformer during back-propagation.

For more details:

- The last Decoder in the stack passes its output to the Output component which converts it into the final output sentence.
- The Linear layer projects the Decoder vector into Word Scores, with a score value for each unique word in the target vocabulary, at each position in the sentence. For instance, if our final output sentence has 7 words and the target Spanish vocabulary has 10000 unique words, we generate 10000 score values for each of those 7 words. The score values indicate the likelihood of occurrence for each word in the vocabulary in that position of the sentence.
- The Softmax layer then turns those scores into probabilities (which add up to 1.0). In each position, we find the index for the word with the highest probability, and then map that index to the corresponding word in the vocabulary. Those words then form the output sequence of the Transformer.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*dQTK3oeYqOBUDVgNktSpCw.png" >
  <br>
  <i>Generate Output in Training the Transformer model</i>
</p>

During training, we use a loss function such as cross-entropy loss to compare the generated output probability distribution to the target sequence. The probability distribution gives the probability of each word occurring in that position.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*50j_urNlRnctglsXCRoeiw.png" >
  <br>
  <i>Compare output with target to calculate loss</i>
</p>

Let’s assume our target vocabulary contains just four words. Our goal is to produce a probability distribution that matches our expected target sequence “De nada END”.

This means that the probability distribution for the first word-position should have a probability of 1 for “De” with probabilities for all other words in the vocabulary being 0. Similarly, “nada” and “END” should have a probability of 1 for the second and third word-positions respectively.

As usual, the loss is used to compute gradients to train the Transformer via backpropagation.

## **4. Inferencing**

During Inference, we have only the input sequence and don’t have the target sequence to pass as input to the Decoder. The goal of the Transformer is to produce the target sequence from the input sequence alone.

So, like in a Seq2Seq model, we generate the output in a loop and feed the output sequence from the previous timestep to the Decoder in the next timestep until we come across an end-of-sentence token.

The difference from the Seq2Seq model is that, at each timestep, we re-feed the entire output sequence generated thus far, rather than just the last word.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*-uvybwr8xULd3ug9ZwcSaQ.png" >
  <br>
  <i>Inference with the Transformer model</i>
</p>

The flow of data during Inference is:

1. The input sequence is converted into Embeddings (with Position Encoding) and fed to the Encoder.
2. The stack of Encoders processes this and produces an encoded representation of the input sequence.
3. Instead of the target sequence, we use an empty sequence with only a start-of-sentence token. This is converted into Embeddings (with Position Encoding) and fed to the Decoder.
4. The stack of Decoders processes this along with the Encoder stack’s encoded representation to produce an encoded representation of the target sequence.
5. The Output layer converts it into word probabilities and produces an output sequence.
6. We take the last word of the output sequence as the predicted word. That word is now filled into the second position of our Decoder input sequence, which now contains a start-of-sentence token and the first word.
7. Go back to step #3. As before, feed the new Decoder sequence into the model. Then take the second word of the output and append it to the Decoder sequence. Repeat this until it predicts an end-of-sentence token. Note that since the Encoder sequence does not change for each iteration, we do not have to repeat steps #1 and #2 each time.

## **5. Transformer challenges**

The vanilla Transformer model helps overcome the RNN model’s shortcomings but it has its own challenges:

- Computational complexity - Transformers are significantly more computationally intensive than RNNs due to the self-attention layers. This makes them more difficult to train on larger datasets.

- Transformers are very sensitive to the quality and quantity of the training data. If the training data is limited or biased, model performance may be adversely affected. This can be a challenge in situations where data is scarce or difficult to obtain.

- Limited context dependency: While Transformers have successfully modeled long-range dependencies in language, they can still struggle with understanding very long text sequences, particularly those that contain multiple levels of abstraction.

- Ethical concerns: Transformers are very good at generating text that is indistinguishable from human-generated text. This can be a problem when the model is used to generate fake news or other types of misleading content.

- ...

## **6. References**

[1] [Paper "Attention Is All You Need"](https://arxiv.org/abs/1706.03762)

[2] [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

[3] [INTUITIVE TRANSFORMERS SERIES NLP](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452)

[4] [Illustrated Guide to Transformers- Step by Step Explanation](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0)
