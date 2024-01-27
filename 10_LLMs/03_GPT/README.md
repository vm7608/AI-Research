# **Generative Pre-trained Transformer (GPT)**

## **1. Introduction**

The first paper from OpenAI on the GPT models is called `Improving Language Understanding by Generative Pre-Training`. Thus, the term GPT has been derived from Generative Pre-Training Transformer.

`GPT (Generative Pre-trained Transformer)` is a state-of-the-art NLP model that has revolutionized the field of AI. Developed by OpenAI, GPT is part of the Transformer architecture family, and it is specifically designed for tasks related to understanding and generating human language.

GPT models are pre-trained on vast amounts of text data and can then be fine-tuned for various language-related tasks. GPT models are known for their ability to generate human-like text and have found applications in chatbots, content generation, and many other areas of natural language understanding and generation.

The GPT model has seen multiple versions, each one more powerful and capable than the last. The first version of GPT was released in 2018, and the latest version, GPT-4, was released on 2023. The following table summarizes the main versions of GPT models:

| **GPT Version**  | **Introduction Year** | **Number of Parameters** | **Paper** | **Achievements** |
|-------------------|------------------------|--------------------------|----------|-------------------|
| GPT-1 | 2018 | 117 million | [Improving Language Understanding by Generative Pre-training](https://paperswithcode.com/paper/improving-language-understanding-by) | Coherent sentence and paragraph generation |
| GPT-2 | 2019 | 1.5 billion | [Language Models are Unsupervised Multitask Learners](https://paperswithcode.com/paper/language-models-are-unsupervised-multitask) | Human-like text generation, various tasks |
| GPT-3 | 2020 | 175 billion | [Language Models are Few-Shot Learners](https://paperswithcode.com/paper/language-models-are-few-shot-learners) | Strong zero-shot and few-shot learning |
| GPT-4 | 2023 | 1 trillion (not official) | [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) | Multimodal (Image and text); Human-level performance on professional benchmarks, academic tasks, and ethical considerations |

GPT models have a wide range of applications in various fields. Here are some of the main applications of GPT models:

- `Content creation:` GPT models can be used to create high-quality content for various purposes, such as writing articles, generating product descriptions, and creating social media posts.
- `Customer service:` GPT models can be used to automate customer service responses by generating human-like text to answer common queries, thus freeing up human customer support agents for more complex issues.
- `Chatbots:` GPT models can be used to power chatbots that can converse with users in a natural language, providing assistance, and answering questions.
- `Language translation:` GPT models can be used to translate text from one language to another.
- `Personal assistants:` GPT models can be used to create personal assistants that can perform tasks such as scheduling appointments, sending emails, and making phone calls.
- ...

These are just a few examples of the many applications of GPT models. As GPT models continue to evolve and improve, we can expect to see even more innovative applications in the future.

## **2. GPT architecture overview**

In general, all GPT models are based on the Transformer architecture, which was first introduced in 2017 by Google. Each version of GPT have some differences, but they all share the same basic architecture. So, in this section, we will discuss the basic architecture of the GPT model and with each version of GPT, we will only consider the differences in architecture and training process.

The following diagram shows the basic architecture of the GPT model:

<p align="center">
  <img src="https://i.imgur.com/c4Z6PG8.png">
  <br>
  <i>Basic architecture of the GPT model</i>
</p>

`GPT model use a stack of only Transformer decoder block with masked self-attention to train the model`. Each version of GPT have a different number of decoder and additional layer, but the basic component is the same as above image.

In short, the input sequence is fed into the model, and the model generates the next word in the sequence. This process is repeated until the desired length is reached. The sequence is passed through a stack of Transformer decoder blocks, each of which consists of a multi-head self-attention layer and a feed-forward neural network. The output of the last decoder block is passed through a linear layer and a softmax activation function to generate the next word in the sequence. The detail intuition of this has been discussed in the Transformer section.

The following image is the detail illustration of the GPT-3 architecture with the configuration of:

- Word embedding size: 12888
- Number of decoder: 96
- Number of self-attention head in each decoder: 96
- Dimension of self-attention head: 128
- Window context size: 2048
- Vocabulary size: 50257

For easily view the image, you can click [here](https://www.canva.com/design/DAFv-ofmLnE/ZepETiJaPa9vTSwUtDyDog/edit?utm_content=DAFv-ofmLnE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).

<p align="center">
  <img src="https://live.staticflickr.com/65535/53233539339_7fbb13abeb_b.jpg" >
  <br>
  <i>Detail illustration of the GPT-3 architecture</i>
</p>

The below table is the detail of implementation of each version of GPT model:

| Model | Word Embedding Size | Number of Decoder Blocks | Number of Self-Attention Heads in the Decoder | Dimension of Self-Attention Head | Context Window Size | Vocabulary Size | Training Dataset | Number of Parameters |
|-------|---------------------|-------------------------|--------------------------------------------|----------------------------------|---------------------------------------|----------------|-----------------|-----------|
| GPT-1 | 768      | 12       | 12        | 64   | 512  | 40,000   | BookCorpus        | 117M   |
| GPT-2 | 768      | 12-48    | 12-48     | 64   | 1024 | 50,257   | WebText           | 1.5B |
| GPT-3 | 12888    | 96       | 96        | 128  | 2048 | Unknown     | CommonCrawl, WebText, English Wikipedia, and two books corpora (Books1 and Books2) | 175B  |
| GPT-4 | Unknown  | Unknown  | Unknown   | Unknown   | 32,768   | Unknown        | Unknown         | In trillions (*)       |

(*) Not official.

## **2. GPT-1**

### **2.1. Learning Objectives and Concepts**

GPT-1 uses a `12-layer decoder-only transformer` framework with `masked self-attention` for training the language model. The GPT model’s architecture largely remained the same as it was in the original work on transformers. With the help of masking, the language model objective is achieved whereby the model doesn’t have access to subsequent words to the right of the current word.

Training a GPT model consists of 2 stages:

1. Pre-training, learning a high-capacity language model on a huge corpus of text.
2. Fine-tuning, where the model is adapted to a discriminative task with labeled data.

#### **2.1.1. Unsupervised pre-training objective**

For unsupervised learning, given an unsupervised corpus of tokens $`\mathcal{U} = {u_1, u_2, ..., u_n}`$, the model is trained to maximize the following log-likelihood:

<p align="center">
  <img src="https://live.staticflickr.com/65535/53226193405_ca433016fb_o.png">
  <br>
  <i>Unsupervised pre-training objective</i>
</p>

Where:

- $`\theta`$ are the parameters of the model.
- $`k`$ is the context window size.
- The conditional probability $`P`$ is modeled using neural network with parameters $`\theta`$.

These parameters trained using stochastic gradient descent.

A multi-layer Transformer decoder model is used, which applies a multi-headed self-attention operation over the input context tokens followed by position-wise feed-forward layers to produce an output distribution over target tokens:

<p align="center">
  <img src="https://live.staticflickr.com/65535/53225687331_d0796540f4_o.png">
  <br>
  <i>Unsupervised pre-training</i>
</p>

Where:

- $`\mathcal{U} = (u_{-k}, ..., u_{-1})`$ is the context vector of tokens
- $`W_e`$ is the token embedding matrix and $`W_p`$ is the position embedding matrix.
- n is the number of layers, $`l`$ is the number of layers in each block and $`k`$ is the context window size.

#### **2.1.2. Supervised fine-tuning objective**

After the model is trained with the above objective, the parameters are adapted to the supervised target task. Now, a labeled dataset $`\mathcal{C}`$ is used where each instance consists of a sequence of input tokens, $`x = {x_1, x_2, ..., x_n}`$ along with a label y. The inputs are passed through the pre-trained model to get the final transformer’s block’s activation $`h^{m}_{l}`$ which is then fed into an additional linear output layer with parameters $`W_y`$ to estimate the conditional probability of the output token $`y`$ given the input sequence $`x`$:

<p align="center">
  <img src="https://editor.analyticsvidhya.com/uploads/78654for2.PNG">
  <br>
  <i>Supervised fine-tuning objective</i>
</p>

This gives us the following objective to maximize:

<p align="center">
  <img src="https://editor.analyticsvidhya.com/uploads/40089for3.PNG">
  <br>
  <i>Supervised fine-tuning objective</i>
</p>

To improve the generalization and enable faster convergence, rather than just maximizing $`L_{2}`$, the authors added an `auxiliary learning objective` for supervised fine-tuning to get better generalisation and faster convergence. The modified training objective was stated as:

<p align="center">
  <img src="https://editor.analyticsvidhya.com/uploads/74132for55.PNG">
  <br>
  <i>Supervised fine-tuning objective</i>
</p>

Where $`L_{1}(C)`$ was the auxiliary objective of learning language model and $`\lambda = 0.5`$ was the weight given to this secondary learning objective.

In short, supervised fine-tuning is achieved by adding a linear and a softmax layer to the transformer model to obtain the task labels for downstream tasks.

#### **2.1.3. Task-Specific Input Transformations**

As mentioned above, we can directly fine-tune the model for tasks like text classification. However, tasks like textual entailment, question answering, etc., that have structured inputs require task-specific customization.

To make minimal adjustments to the model’s architecture during fine-tuning, inputs to the particular downstream tasks are transformed into ordered sequences. The tokens are rearranged as follows:

- Start and end tokens are added to the input sequences.
- A delimiter token is also added between different parts of the example to pass the input as an ordered sequence.

For tasks like question-answering (QA), multiple choice questions (MCQs), etc, multiple sequences are sent for each example.

<p align="center">
  <img src="https://editor.analyticsvidhya.com/uploads/66283TEXT%20GEN.PNG">
  <br>
  <i>(Left) Generative Pre-training transformer architecture and training objectives used in this work input (Right)</i>
</p>

### **2.2. Dataset**

GPT-1 language model was trained using the `BooksCorpus dataset` which consists of about 7000 unpublished books that helped in training the language model on unseen data. This corpus also contained long stretches of contiguous text, which assisted the model in processing long-range dependencies.

### **2.2. Implement details**

#### **2.2.1. For Unsupervised Pre-training**

The following is the details of the Unsupervised Pre-training process:

- GELU (Gaussian Error Linear Units) activation function is used. The main properties of GELU are:
  - `Smoothness`: GELU is a smooth function, meaning it’s continuously differentiable, unlike ReLU, which has a discontinuity at zero. The smoothness property can help with gradient-based optimization during training.
  - `Approximation to Identity`: For small values of x, GELU behaves like the identity function, which allows it to preserve information in the network even for small activations.
  - Non-Linearity: GELU introduces non-linearity to the network, which is essential for capturing complex relationships in data.
  - Sigmoid and Tanh Components: GELU uses a combination of the sigmoid and hyperbolic tangent (tanh) functions, which helps it model negative values and gradients effectively.
  - Normalization: GELU includes a normalization term sqrt(2/pi) to ensure that the outputs have a standard deviation close to 1, which can help stabilize training in deep neural networks.
  - Read more about GELU: [Khám phá activation function Gelu(Transformers)](https://viblo.asia/p/kham-pha-activation-function-gelutransformers-EoW4og0B4ml) and [Deep Learning: GELU (Gaussian Error Linear Unit) Activation Function](https://aaweg-i.medium.com/deep-learning-gelu-gaussian-error-linear-unit-activation-function-56168dd5997)

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*FkNwBrxFa8HxprWVpGIodg.png">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/ReLU_and_GELU.svg/1232px-ReLU_and_GELU.svg.png">
  <br>
  <i>GELU (Gaussian Error Linear Units)</i>
</p>

- Adam optimizer with a learning rate of 2.5e-4 was used. Read more about Adam optimizer [here](https://viblo.asia/p/optimizer-hieu-sau-ve-cac-thuat-toan-toi-uu-gdsgdadam-Qbq5QQ9E5D8).

- `Byte Pair Encoding` vocabulary with 40,000 merges was used. BPE is a simple form of data compression algorithm in which the most common pair of consecutive bytes of data is replaced with a byte that does not occur in that data. Suppose we have data "aaabdaaabac" which needs to be encoded (compressed).
  - The byte pair "aa" occurs most often, so we will replace it with Z as Z does not occur in our data. So we now have "ZabdZabac" where Z = "aa".
  - The next common byte pair is "ab" so replace it with Y. We now have "ZYdZYac" where Z = "aa" and Y = "ab".
  - The only byte pair left is "ac" which appears as just one so we will not encode it.
  - We can use recursive byte pair encoding to encode ZY as X. Our data has now transformed into "XdXac" where X = ZY, Y = "ab", and Z = "aa". It cannot be further compressed as there are no byte pairs appearing more than once.
  - We decompress the data by performing replacements in reverse order.

- The 3072-dimensional state was employed for the position-wise feed-forward layer.

- The 12-layered model used 12 attention heads in each self-attention layer.

- The tokens were encoded into word embeddings using a 768-dimensional state in the model.

- Residual, and embedding dropouts were used for regularisation, with a dropout rate of 0.1. A customized version of L2 regularisation was also used for non-bias weights.

- The model was trained over 100 epochs on mini-batches of size 64 and a sequence length of 512. The model had 117M parameters.

#### **2.2.2. Fine-tuning**

The majority of the hyperparameters from the unsupervised pre-training were used for fine-tuning. For most of the downstream tasks, supervised fine-tuning only required 3 epochs. This demonstrated how much the model had already learned about the language during the pre-training phase. So, a little fine-tuning was enough to adapt.

GPT-1 demonstrated that the language model served as an effective pre-training objective which could aid the model to generalize well. The architecture enabled transfer learning and could perform various NLP tasks with very little need for fine-tuning. This model demonstrated the potency of generative pre-training and provided a path for the development of additional models that could better realize this potential given a larger dataset and more parameters.

## **3. GPT-2**

The developments in GPT-2 model were mostly in terms of using a larger dataset and adding more parameters to the model compare to GPT-1 to learn even stronger language model.

### **3.1. Learning Objectives and Concepts**

`Task Conditioning:` We had seen that training objective of language model is formulated as `P(output|input)`. However, GPT-2 aimed at learning multiple tasks using the same unsupervised model. To achieve that, the learning objective should be modified to `P(output|input, task)`. This modification is known as task conditioning, where the model is expected to produce different output for same input for different tasks.

Some models implement task conditioning at an architectural level where the model is fed both, the input and the task. For language models, the output, input and task, all are sequences of natural language. Thus, task conditioning for language models is performed by providing examples or natural language instructions to the model to perform a task. Task conditioning forms the basis for zero-shot task transfer.

`Zero Shot Learning and Zero Short Task Transfer:` Zero shot learning is a special case of zero shot task transfer where no examples are provided at all and the model understands the task based on the given instruction. Instead of rearranging the sequences, as was done for GPT-1 for fine-tuning, input to GPT-2 was given in a format which expected the model to understand the nature of task and provide answers. This was done to emulate zero-shot task transfer behaviour. E.g. for English to French translation task, the model was given an English sentence followed by the word French and a prompt (:). The model was supposed to understand that it is a translation task and give French counterpart of English sentence.

### **3.2. Dataset**

To create an extensive and good quality dataset the authors scraped the Reddit platform and pulled data from outbound links of high upvoted articles. The resulting dataset called `WebText`, had 40GB of text data from over 8 million documents. This dataset was used for training GPT-2 and was huge compared to Book Corpus dataset used for training GPT-1 model. All Wikipedia articles were removed from WebText as many test sets contain Wikipedia articles.

### **3.3. Model architecture and Implementation Details**

GPT-2 had 1.5 billion parameters, which was 10 times more than GPT-1 (117M parameters). Major differences from GPT-1 were:

- GPT-2 had 48 layers and used 1600 dimensional vectors for word embedding.
- Larger vocabulary of 50,257 tokens was used.
- Larger batch size of 512 and larger context window of 1024 tokens were used.
- Layer normalisation was moved to input of each sub-block and an additional layer normalisation was added after final self-attention block.
- At initialisation, the weight of residual layers was scaled by $`\frac{1}{\sqrt{N}}`$ where N is the number of residual layers.

The authors trained four language models with 117M (same as GPT-1), 345M, 762M and 1.5B (GPT-2) parameters. Each subsequent model had lower perplexity than previous one. This established that the perplexity of language models on same dataset decreases with an increase in the number of parameters. Also, the model with the highest number of parameters performed better on every downstream task.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53232357837_f1da5d476e_o.png">
  <br>
  <i>Details of variants of the GPT-2 model</i>
</p>

## **4. GPT-3**

In the try to build very strong and powerful language models which would need no fine-tuning and only few demonstrations to understand tasks and perform them, Open AI built the GPT-3 model with 175 billion parameters which had 100 times more parameters than GPT-2.

Due to large number of parameters and extensive dataset GPT-3 has been trained on, it performs well on downstream NLP tasks in zero-shot and few-shot setting. Owing to its large capacity, it has capabilities like writing articles which are hard to distinguish from ones written by humans. It can also perform on-the-fly tasks on which it was never explicitly trained on, like summing up numbers, writing SQL queries and codes, etc.

### **4.1. Learning Objectives and Concepts**

- `In-context learning`: LLMs develop pattern recognition and other skills using the text data they are trained on. While learning the primary objective of predicting the next word given context words, the language models also start recognising patterns in data which help them minimise the loss for language modelling task. Later, this ability helps the model during zero-shot task transfer. When presented with few examples and/or a description of what it needs to do, the language models matches the pattern of the examples with what it had learnt in past for similar data and uses that knowledge to perform the tasks. This is a powerful capability of large language models which increases with the increase in the number of parameters of the model.

- `Few-shot, one-shot and zero-shot setting`: few, one and zero-shot settings are specialised cases of zero-shot task transfer. In few-shot setting, the model is provided with task description and as many examples as fit into the context window of model. In one-shot setting the model is provided exactly one example and in zero-shot setting no example is provided. With increase in capacity of model, few, one and zero-shot capability of model also improves.

### **4.2. Dataset**

GPT-3 was trained on a mix of five different corpora, each having certain weight assigned to it. High quality datasets were sampled more often, and model was trained for more than one epoch on them. The five datasets used were Common Crawl, WebText2, Books1, Books2 and Wikipedia.

### **4.3. Model architecture and Implementation Details**

The architecture of GPT-3 is same as GPT-2. Few major differences from GPT-2 are:

- GPT-3 has 96 layers with each layer having 96 attention heads.
- Size of word embeddings was increased to 12888 for GPT-3 from 1600 for GPT-2.
- Context window size was increased from 1024 for GPT-2 to 2048 tokens for GPT-3.
- Adam optimiser was used with β_1 = 0.9, β_2 = 0.95 and ε = 10^(-8).
- Alternating dense and locally banded sparse attention patterns were used.

There are several GPT-3 examples and variants in terms of:

- Sizes (Parameters and Layers)
- Architectures
- Learning hyper-parameters (batch size in tokens and learning rate)

Here are the details of the different variants of GPT-3 model:

<p align="center">
  <img src="https://www.sigmoid.com/wp-content/uploads/2022/12/sigmoid_gpt-3_details.png">
  <br>
  <i>Details of variants of the GPT-3 model</i>
</p>

### **4.4. Reinforcement Learning from Human Feedback**

GPT-3 was aligned to human references using RLHF, this process consists of 3 distinct steps:

1. `Supervised fine-tuning` – A pre-trained language model is fine-tuned on a relatively small amount of demonstration data curated by labelers, to learn a supervised policy that generates outputs from a selected list of prompts. This represents the baseline model.

2. `“Mimic human preferences”` – Labelers are asked to vote on a relatively large number of the fine-tuned model outputs, this way creating a new dataset consisting of comparison data. A new model is trained on this dataset. This is referred to as the reward model (RM).

3. `Proximal Policy Optimization (PPO)` – The reward model is used to align and improve the fine-tuned model. The outcome of this step is the so-called policy model.

Step 1 takes place only once, while steps 2 and 3 can be iterated continuously: more comparison data is collected on the current best policy model, which is used to train a new reward model and then a new policy.

The following image illustrates the above process:

<p align="center">
  <img src="https://images.openai.com/blob/cf717bdb-0c8c-428a-b82b-3c3add87a600/ChatGPT_Diagram.svg?width=10&height=10&quality=50" >
  <br>
  <i>Align GPT-3 with RLHF</i>
</p>

## **5. GPT-4**

### **5.1. Abstract and Introduction**

This is the summary about the GPT-4 Technical Report:

- GPT-4 is a large and a multimodal language model which mean it can receive input including images and text and output will return the result as text.
- The GPT-4 model performed comparable to humans on a number of different professional and academic benchmarks. In particular, it passed the law exam simulation test with a score in the top 10% highest scores.
- GPT-4 is still a Transformer-based language model and is trained to predict and generate the next words in the document.
- GPT-4 does perform a post-training alignment process, which is simply the process of fine-tuning the model to increase the realism of the answer as well as comply with predefined behavior.
- An important part of this project is to develop the infrastructure and optimization methods to be able to predict the performance of the model (or the value of the loss function) with different scales. It is called scaling laws, which means the relationship between input quantities such as the size of the model, the size of the dataset, the calculation time to the accuracy or loss value of the model. Based on these scaling laws, it is possible to interpolate the necessary parameters of the amount of model size data or computational resources needed to achieve the desired results.

The is the summary about some of the more outstanding features of the GPT-4 compared to previous models

- GPT-4 is a large, multimodal language model that can receive input including images and text and output will return the result as text.
- Improved natural language understanding and generation.
- High performance on traditional NLP benchmarks.
- Strong performance in multiple languages

### **5.3. Scope and Limitations of this Technical Report**

<p align="center">
  <img src="https://images.viblo.asia/2461844a-449f-4273-8538-826bfce5ad7f.png" >
  <br>
  <i>The GPT-4 Technical report does not share any details about architecture (including model size), hardware, computational resources used for training, dataset construction, training methods, etc.</i>
</p>

### **5.4. Predictable Scaling**

The main idea of `Scaling laws`:

- Scaling laws helps us determine the correlation between the value of loss or accuracy with input factors such as dataset size, model size, amount of computational resources... This is especially useful when training large models such as LLMs.

- OpenAI said that they have designed the hardware architecture and computing resources in a very reasonable way to be able to accurately predict how much loss value the GPT-4 model will achieve when additional computational resources are added for training. This scaling prediction is done based on models of smaller size and the same task and the same training strategy. You can see in the following picture.

<p align="center">
  <img src="https://images.viblo.asia/0604387f-55f7-43e7-a292-e40de4c47858.png" >
  <br>
  <i>Scaling laws helps us determine the correlation between the value of loss or accuracy with input factors such as dataset size, model size, amount of computational resources... This is especially useful when training large models such as LLMs.</i>
</p>

The above mean that OpenAI have designed infrastructure and optimization strategies to satisfy scaling laws. The dashed line is the line fitted according to scaling laws (Prediction) with the dots being the loss value obtained when training smaller models and lower computational resources (Observed). This predict is accurate on very smaller model. Which mean they can predict how much computational resources will be needed to achive an accuray. This is very important cuz training LLMs is costly.

### **5.5. Some results of GPT-4**

#### **Pass the exam**

Among the many results highlighted by OpenAI, what immediately stands out is GPT-4's performance on a series of standardized tests. For example, GPT-4 scored in the top 10% of scores on the simulated U.S. bar exam, while GPT-3.5 scored in the bottom 10% of scores.

<p align="center">
  <img src="https://images.viblo.asia/4841efc8-07f7-4b92-b58b-d355249ddc33.png" >
  <br>
  <i>Pass the exam</i>
</p>

#### **Able to reply on image information**

<p align="center">
  <img src="https://images.viblo.asia/a4cb9682-1d46-49b9-a4f2-bad78d060b7c.png" >
  <img src="https://images.viblo.asia/185709f2-ba99-48f7-af07-dc71bf8ea286.png" >
  <br>
  <i>Reply on image information</i>
</p>

#### **Better inference than ChatGPT**

The answers of GPT-4 and ChatGPT can be compared to the same problem.

<p align="center">
  <img src="https://images.viblo.asia/995b6f41-02ed-4cc4-906d-450f8b2d6e8c.png" >
  <img src="https://images.viblo.asia/c22e3fa1-98f6-4575-ad8f-8b8c8e539405.png" >
  <br>
  <i>Reply on image information</i>
</p>

#### **Larger context window and longer text produced**

GPT-4 can receive and generate up to 25,000 words of text, much more than ChatGPT's limit of about 3,000 words. It can handle more complex and detailed prompts, and produce more extensive text. In the example below, GPT-4 was given the entire Wikipedia article about AI and it answered specific question correctly.

<p align="center">
  <img src="https://images.viblo.asia/full/ae562738-5988-4ec9-b3ba-de0d88db24cc.png" >
  <br>
  <i>Example on very long article</i>
</p>

### **5.6. Limitations of GPT-4**

There are some limitations that OpenAI has identified in GPT-4, including:

- May produce false results and cause hallucinations (not fully reliable).
- The GPT-4 model is still limited to context windows.
- GPT-4 does not automatically learn from past experiences.
- Caution should be exercised when using the output of GPT-4, especially in contexts where reliability is important.
- GPT-4's capabilities and limitations create new and significant safety challenges that require careful study and mitigation.

## **6. OpenAI API**

OpenAI has released an API that allows users to access, fine-tune, and deploy GPT-3 models. They also provide some endpoints API for audio, chat, completions, embedding, etc. All the stuff is available at [OpenAI API](https://platform.openai.com/docs/api-reference).

## **7. References**

[1] [The Journey of Open AI GPT models](https://medium.com/walmartglobaltech/the-journey-of-open-ai-gpt-models-32d95b7b7fb2)

[2] [The GPT-3 Architecture, on a Napkin](https://dugas.ch/artificial_curiosity/GPT_architecture.html)

[3] [The Illustrated GPT-2 (Visualizing Transformer Language Models)](http://jalammar.github.io/illustrated-gpt2/)

[4] [How GPT3 Works - Visualizations and Animations](http://jalammar.github.io/how-gpt3-works-visualizations-animations/)
