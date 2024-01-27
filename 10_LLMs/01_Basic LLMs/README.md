# **LARGE LANGUAGE MODELS**

## **1. Overview and Introduction**

### **1.1. Language Models (LMs)**

A language model is a probabilistic model of a natural language that can generate probabilities of a series of words, based on text corpora in one or multiple languages it was trained on. Language models are useful for a variety of tasks, including speech recognition (helping prevent predictions of low-probability (e.g. nonsense) sequences), machine translation, natural language generation (generating more human-like text), optical character recognition, handwriting recognition, grammar induction, information retrieval and other.

Language modeling (LM) uses statistical and probabilistic techniques to determine the probability of a given sequence of words occurring in a sentence. Hence, a language model is basically a probability distribution over sequences of words:

```math
P(x^{(t+1)} | x^{(t)}, x^{(t-1)}, ..., x^{(1)})
```

Here, the expression computes the conditional probability distribution where $`x^{(t+1)}`$ can be any word in the vocabulary.

Language models generate probabilities by learning from one or more text corpus. A text corpus is a language resource consisting of a large and structured set of texts in one or more languages.

One of the earliest approaches for building a language model is based on the `n-gram`. An `n-gram` is a contiguous sequence of n items from a given text sample. Here, the model assumes that the probability of the next word in a sequence depends only on a fixed-size window of previous words:

<p align="center">
  <img src="https://www.baeldung.com/wp-content/uploads/sites/4/2023/05/Language-Model-N-gram.jpg" >
  <br>
  <i>N-gram</i>
</p>

However, n-gram language models have been largely superseded by `neural language models`. These models make use of continuous representations or embeddings of words to make their predictions:

<p align="center">
  <img src="https://www.baeldung.com/wp-content/uploads/sites/4/2023/05/Language-Model-Neural-Network.jpg" >
  <br>
  <i>Neural Language Model</i>
</p>

Basically, neural networks represent words distributedly as a non-linear combination of weights. Hence, it can avoid the curse of dimensionality in language modeling. There have been several neural network architectures proposed for language modeling such as Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and Transformer.

### **1.2. Large Language Models (LLMs)**

`Large Language Models (LLMs)` are basically `neural language models working at a larger scale`. A large language model consists of a neural network with possibly billions of parameters. Moreover, it’s typically trained on vast quantities of unlabeled text, possibly running into hundreds of billions of words.

However, due to sufficient training on a large set of data and an enormous parameter count, these models can capture much of the syntax and semantics of the human language. Hence, they become capable of finer skills over a wide range of tasks in computational linguistics.

This is quite a departure from the earlier approach in NLP applications, where specialized language models were trained to perform specific tasks. On the contrary, researchers have observed many emergent abilities in the LLMs, abilities that they were never trained for.

For instance, LLMs have been shown to perform multi-step arithmetic, unscramble a word’s letters, and identify offensive content in spoken languages. ChatGPT, a popular chatbot built on top of OpenAPI’s GPT family of LLMs, has cleared professional exams like the US Medical Licensing Exam.

### **1.4. LLMs Application**

Large Language Models (LLMs) have a wide range of applications in natural language processing, including:

- Chatbots and virtual assistants
- Language translation services
- Text summarization
- Sentiment analysis
- Question answering
- Text completion and generation
- Spell checking and correction
- Named entity recognition
- Language modeling for speech synthesis

LLMs are also used in various industries such as healthcare, finance, and e-commerce for tasks such as customer service, fraud detection, and personalized recommendations.

### **1.5. Challenge and Limitations**

Large Language Models (LLMs) have been shown to perform well on a wide range of tasks. However, they are not without their limitations. Some of the main limitations and challenges of Large Language Models (LLMs) include:

- `Speed and cost:` LLMs are computationally expensive when training and deploying, which can make them difficult to scale. LLMs also require large amounts of training data, and the quality of the data can impact the accuracy of the model.
- `Hallucinations:` LLMs can generate text that is not grounded in reality, which can be problematic in certain applications.
- `Lack of transparency:` LLMs can be difficult to interpret, and it can be challenging to understand how they arrive at their outputs.
- `Potential bias:` LLMs can perpetuate biases that exist in the training data, which can lead to unfair or discriminatory outcomes.
- `Data privacy:` LLMs require large amounts of data to train, which can raise concerns about data privacy and security.
- `Inconsistent accuracy:` Each LLM has been trained for a specific purpose, and the best model will depend on the specific NLP task at hand. Therefore, it is important to note that LLMs can have inconsistent accuracy. LLMs aslo can struggle with understanding context, which can lead to inaccurate or irrelevant outputs.
- `Limited controllability:` LLMs can be difficult to control, and it can be challenging to ensure that they generate outputs that align with specific goals or values.
- `Ethical concerns`:Generating misinformation: LLMs can generate text that is misleading or false, LLMs can raise ethical concerns, particularly around issues of bias, privacy, and accountability.

## **2. General Architecture of LLMs**

### **2.1. Earlier Architecture of LLMs**

When it started, LLMs were largely created using self-supervised learning algorithms. Self-supervised learning refers to the `processing of unlabeled data to obtain useful representations` that can help with downstream learning tasks. Quite often, self-supervised learning algorithms use a model based on an artificial neural network (ANN) and the most widely used architecture for LLMs were the Recurrent Neural Network (RNN):

However, `RNNs are not capable of capturing long-term dependencies in the input sequence`. This is because the gradient of the loss function decays exponentially with time. Hence, the RNNs are unable to learn long-term dependencies in the input sequence.

<p align="center">
  <img src="https://www.baeldung.com/wp-content/uploads/sites/4/2023/05/Vanishing-Gradient-Problem.jpg" >
  <img src="https://www.researchgate.net/publication/343462095/figure/fig4/AS:924083218808832@1597330016103/Computation-wise-comparison-of-RNN-LSTM-and-GRU-nodes.png" >
  <br>
  <i>Vanishing Gradient Problem</i>
</p>

`LSTM` was developed to address the vanishing gradient problem in RNNs. It introduced a more complex structure with memory cells, input and output gates, and forget gates. LSTMs can capture and retain information over longer sequences, making them more suitable for tasks with extended dependencies and time series data. However, LSTMs are computationally expensive and require a lot of training data.

`GRUs` are a more streamlined variation of RNNs designed to simplify the LSTM architecture. They retain the ability to capture long-term dependencies by using gating mechanisms but have a reduced number of gates and computations. GRUs are computationally efficient and, in some cases, perform as well as LSTMs, making them a good choice for various sequential data tasks.

### **2.2. Transformer Model**

The introduction of transformers by the Google Brain team in 2017 is perhaps one of the most important inflection points in the history of LLMs. A transformer is a deep learning model that adopts the self-attention mechanism and processes the entire input all at once. The transformer architecture is based on the encoder-decoder architecture, as shown in the following diagram:

<p align="center">
  <img src="https://miro.medium.com/max/1280/1*5yxKYbi_K2NihsW6Z0Tq-Q.png" >
  <br>
  <i>Transformer architecture</i>
</p>

The Transformer model introduced two significant improvements compared to previous models

1. `Parallelization and Efficiency`: The most substantial improvement in the Transformer model is its ability to parallelize the processing of input sequences. In RNNs, processing elements in a sequence was inherently sequential, which limited parallelization and made training slow. Transformers, on the other hand, use self-attention mechanisms that allow for parallel computation across all elements in the input sequence. This parallelization significantly speeds up training and inference.

2. `Long-Range Dependencies`: Transformers is better at capturing long-range dependencies in sequences, which is challenging for RNNs. The self-attention mechanism enables each token to attend to all other tokens in the input sequence, regardless of their distance. This makes Transformer highly effective for tasks involving long-distance contextual understanding, such as machine translation and language modeling.

The detail of Transformer we have discussed in the previous article [here](https://github.com/vm7608/AI-Training-2/tree/main/04_Generative_AI/00_Transformer).

## **3. Application Techniques for LLMs**

### **3.1. Prompting and prompt engineering**

`Prompt` is the textual context that we feed into the model to help guide its generation of responses. Common types of prompts include conversational cues like "Let's discuss...", descriptive contexts, questions, continuations of previous dialog, etc. Prompts can be as brief as a few words or multiple paragraphs.

The output text of the model is known as the `completion`. The act of generating text above call `inference`. The full amount of text or the memory that is available to use for the prompt is called the `context window`.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214511691_f5b8076be5_o.png" >
  <br>
  <i>Promt</i>
</p>

Maybe sometimes, the model performs well on the first try. Still, we'll frequently encounter situations where the model doesn't produce the outcome that we want on the first try. We may have to revise the language in our prompt or the way that it's written several times to get the model to behave in the way that we want. This work to develop and improve the prompt is called `prompt engineering`.

`Prompt engineering` refers to the process of carefully crafting textual prompts to elicit targeted behaviors from LLMs. The goal of prompt engineering is developing textual contexts that maximize the benefits of language models while mitigating risks like toxic, factually incorrect, or unaligned responses. It's an important technique for applying LLMs safely and ensuring user experiences meet expectations. Proper prompt design upfront helps avoid potential downstream issues.

However, one powerful strategy to get the model to produce better outcomes is to include examples of the task that we want the model to carry out inside the prompt. Providing examples inside the context window is called `in-context learning`. There are the following types of in-context learning:

- `Zero-shot inference`: The model is given a prompt that describes the task, but no examples of the task. The model is expected to perform the task without any examples.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214901024_ba4fe7f7c2_o.png" >
  <br>
  <i>Zero-shot inference</i>
</p>

- `One-shot inference`: For smaller models, zero-shot inference may not return a good result. So we have one-shot inference where the model is given a prompt that describes the task and one complete example of the task. The model is expected to perform the task with only one example.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214511696_4eb3e8eb2c_o.png" >
  <br>
  <i>One-shot inference</i>
</p>

- `Few-shot inference`: For even smaller models that fail to perform the task with one example, we have few-shot inference. The model is given a prompt that describes the task and a few examples of the task.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214901049_a24286dba0_b.jpg" >
  <br>
  <i>Summary of in-context learning</i>
</p>

A drawback of in-context learning is that out context window have a limit amount of token so that we can't include too many examples. And also, in-context learning may not always work well for small models.

Generally, if we find that our model isn't performing well when including five or six examples, we should try fine-tuning our model instead. Fine-tuning performs additional training on the model using new data to make it more capable of the task we want it to perform.

### **3.2. Generative configuration parameters for inference**

In the LLMs model, we have some parameters that we can use to control the model's behavior. These parameters are called `generative configuration parameters`. Each model has a set of configuration parameters that can influence the model's output during inference.

Note that these are different from the training parameters which are learned during training. Instead, `these configuration parameters are invoked at inference time` and give us control over things like the maximum number of tokens in the completion, and how creative the output is.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215063660_90c85c7e80_o.png" >
  <br>
  <i>Configuration parameters</i>
</p>

`Max new tokens` is the maximum number of tokens that the model can generate in the completion. This is useful for limiting the length of the completion. Note that max new tokens is not a hard limit on the number of tokens in the completion. The model may generate fewer tokens than the max new tokens value if it reaches a stopping condition.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53213670217_e29315c14b_o.png" >
  <br>
  <i>Max new tokens</i>
</p>

The output from the transformer's softmax layer is a probability distribution across the entire vocabulary. Most LLMs by default will operate with `greedy decoding` where the model will always choose the word with the highest probability. This method can work very well for short text generation but it can lead to repeated words or sequences of words.

If we want to generate text that's more natural, more creative and avoids repeating words, we need to use some other controls. `Random sampling` is the easiest way to introduce some variability. Instead of selecting the most probable word every time, with random sampling, the model chooses an output word randomly using the probability distribution.

For the below example, the word "banana" has a probability score of 0.02. With random sampling, this equates to a 2% chance that this word will be selected. In this way, we reduce the possibility of repeated words. However, depending on the setting, the output may be too creative, producing wrong topics or nonsense words. In some implementations, we may need to disable greedy and enable random sampling explicitly.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53213684897_2ce6f82b9a_o.png" >
  <br>
  <i>Greedy vs Random sampling</i>
</p>

To control the LLMs ouput creativity, we have 3 parameters called `Top k`, `Top p` and `Temperature`.

`Top k` focuses on a small subset of most likely tokens. At each step, the model considers only the top k highest probability tokens. The model then selects from these k options using the probability weighting. This method can help the model have some randomness while reducing the selection of highly improbable completion words. This makes our text generation more likely to sound reasonable and make sense.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53213684887_f04280c5b0_o.png" >
  <br>
  <i>Top k</i>
</p>

We can also use the `Top p` approach to control random sampling. This method restricts the selection to predictions whose combined probabilities don't go beyond a specified threshold, denoted as `p`. For instance, if we set p to 0.3, we accumulate probabilities from the most probable to the least until we reach the 0.3 limit. In this example, the choices available are "cake" and "donut" because their individual probabilities, 0.2 and 0.1, sum up to 0.3. The model then use a random probability-based selection method to pick one of these tokens.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215079035_2a65f1ebd9_o.png" >
  <br>
  <i>Top p</i>
</p>

One more parameter to control the randomness of the model output is `temperature`. This parameter influences the shape of the probability distribution that the model calculates for the next token. The `higher the temperature, the higher the randomness`, and the `lower the temperature, the lower the randomness`. The temperature value is a scaling factor that's applied within the final softmax layer of the model that impacts the shape of the probability distribution of the next token. In contrast to the top k and top p parameters, changing the temperature actually alters the predictions that the model will make.

- If we choose a low value of temperature (< 1), the softmax layer produces a probability distribution that's highly concentrated, with most of the probability focused on a small set of words. In this case, the word "cake" stands out. The model selects from this distribution through random sampling, resulting in text that's less random and closely resembles the most probable word sequences the model learned during training.

- If we set the temperature to a higher value (> 1), the model generates a more evenly spread and broader probability distribution for the next token. Unlike above, where probability is spread more uniformly among tokens, this leads to text that's more random and diverse compared to a lower-temperature setting. This can be useful for generating text that sounds more creative.

- If we leave the temperature value equal to one, this will leave the softmax function as default and the unaltered probability distribution will be used.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215079040_ab0294e98b_b.jpg" >
  <br>
  <i>Temperature examples</i>
</p>

### **3.3. Generative AI project lifecycle**

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214618491_b84df1562c_o.png" >
  <br>
  <i>Generative AI project lifecycle</i>
</p>

The basic Project lifecycle of a Generative AI deals with 4 core principles:

- `Define Scope`: The initial and crucial step in any project is to precisely and narrowly define the scope. The capabilities of Large Language Models (LLMs) can vary based on their size and architecture. It's essential to determine the specific function the LLM will have in a given application.

- `Select model`: Next, we have to decide whether to train our own model from scratch or work with an existing base model. In general, we should start with an existing model, although there are some cases where we may have to train a model from scratch.

- `Adapt and align model`: The next step is to assess model's performance and carry out additional training if needed.
  - We should start by trying `prompt engineering` and `in-context learning` to enhance performance.
  - If the model doesn't meet the performance requirements even with one or a few short inference, we can try fine-tuning our model.
  - When models become more capable, it's important to ensure that they behave well and be aligned with human preferences. Here, we can perform an additional fine-tuning technique called `reinforcement learning with human feedback`, which can help to make sure that model behaves well.
  - Finally, evaluation using metrics and benchmarks to assess the model's performance and alignment with preferences is crucial.

- `Application integration`: when our model is good perform and well aligned, we can deploy or integrate it in an application. At this stage, it's to optimize our model to ensure of efficient compute resources and good experience for the users.

### **3.4. Pre-training large language models**

#### **3.4.1. Pre-training LLMs at a high level**

The initial training process for LLMs is often referred to as `pre-training`. LLMs encode a deep statistical representation of language. This understanding is developed during the model's pre-training phase when the model learns from vast amounts of unstructured textual data. This can be gigabytes, terabytes, and even petabytes of text. This data is pulled from many sources, including scrapes off the Internet and corpora of texts that have been assembled specifically for training language models.

In this self-supervised learning step, the model studies patterns and relationships within language. These patterns are used to achieve the model's training goal, which based on the model's architecture. During pre-training, the model weights are updated to minimize the loss of the training objective. Pre-training requires a large amount of compute and the use of GPUs.

Note that when we scrape training data from public sites such as the Internet, we often need to process the data to increase quality, address bias, and remove other harmful content. As a result of this data quality curation, often only 1-3% of tokens are used for pre-training.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214984308_c255a486d8_b.jpg" >
  <br>
  <i>LLM pre-training at a high level view</i>
</p>

#### **3.4.2. Transformer model types**

There are 3 main variance of the transformer model: `encoder-only`, `encoder-decoder models`, and `decode-only`. Each of these is trained on a different objective, and so learns how to carry out different tasks.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214984298_f977d52c0c_o.png" >
  <br>
  <i>Transformer model types</i>
</p>

`Encoder-only models` (also known as `Autoencoding models`) are pre-trained using `masked language modeling`. In this approach, tokens within the input sequence are either masked or randomly hidden, and the primary objective during training is to predict these masked tokens, effectively reconstructing the original sentence. This process is often referred to as a `denoising objective`.

Autoencoding models provide bidirectional representations of input sequences, allowing them to understand the context of a token not only from preceding words but also from the entire sequence.Encoder-only models are particularly well-suited for tasks that benefit from this comprehensive context such as sentence classification tasks (e.g., sentiment analysis) and token-level tasks like named entity recognition or word classification. Some well-known examples of an autoencoder model are BERT and RoBERTa.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214984358_f0283ba142_b.jpg" >
  <br>
  <i>Autoencoding models</i>
</p>

`Decoder-only` or `autoregressive models`, which are pre-trained using `causal language modeling`. In this approach, the primary training objective is to predict the next token based on the preceding sequence of tokens. These autoregressive models, which are decoder-based, mask the input sequence and can only consider the input tokens leading up to the token they're currently predicting. They has no knowledge of the end of the sentence.

To make predictions, the model goes through the input sequence step by step, predicting the subsequent token in a unidirectional manner. In contrast to encoder-based architectures, the context in these models is one-sided, meaning it only considers what precedes the token being predicted. By learning from a vast number of examples and predicting the next token repeatedly, these models develop a statistical understanding of language.

Decoder-only models often find application in text generation tasks, and larger decoder-only models exhibit strong zero-shot inference capabilities, performing well across various tasks. Examples of decoder-based autoregressive models include GPT and BLOOM.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215066264_2bf7b13ca8_o.png" >
  <br>
  <i>Autoregressive models</i>
</p>

The `sequence-to-sequence` variation of the transformer model `combines both the encoder and decoder components of the original transformer architecture`.

The exact details of the pre-training objective vary from model to model. A popular sequence-to-sequence model T5, pre-trains the encoder using `span corruption`,  which involves masking random sequences of input tokens.

These masked sequences are then substituted with a special token known as the `Sentinel token` (represented as "x"). Sentinel tokens are unique tokens added to the vocabulary but do not correspond to any actual words in the input text. Subsequently, the decoder's task is to reconstruct the sequences of masked tokens in an autoregressive manner. The output includes the Sentinel token followed by the predicted tokens.

Sequence-to-sequence models, such as T5, are valuable for tasks like translation, summarization, and question-answering, particularly in situations where both input and output are in the form of text data.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214673701_09dca5b6d6_b.jpg" >
  <br>
  <i>Sequence-to-sequence models</i>
</p>

In summary, the different model architectures and their pre-training objectives can be compared as follows:

- `Autoencoding models` are pre-trained using `masked language modeling`. They primarily correspond to the encoder part of the original transformer architecture and are frequently applied in tasks like sentence or token classification.

- `Autoregressive models` are pre-trained using `causal language modeling`. These models utilize the decoder component of the original transformer architecture and are commonly used for text generation.

- `Sequence-to-sequence models` employ both the encoder and decoder components of the original transformer architecture. The specific pre-training objectives may vary between models. For instance, the T5 model is pre-trained using `span corruption`. Sequence-to-sequence models find utility in tasks such as translation, summarization, and question-answering.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214984338_2730a89474_o.png" >
  <br>
  <i>Model architectures and pre-training objectives</i>
</p>

## **4. Fine-tuning LLMs**

Earlier, we see the main drawback of in-context learning is that context window have a limited amount of tokens so we can't include too many examples. And also, in-context learning may not always work well for small models even when we include five or six examples.

Luckily, we have another technique called `fine-tuning` to solve this problem. Fine-tuning performs additional training on the model using new data to make it more capable of the specific task we want it to perform.

### **4.1. Fine-tuning LLMs at a high level**

In contrast to pre-training, where we train the LLMs using vast amounts of unstructured textual data via `self-supervised learning`, `fine-tuning is a supervised learning process`. In fine-tuning, a dataset of labeled examples, consisting of `prompt-completion pairs`, is used to adjust the LLM's weights. This fine-tuning process extends the model's training to enhance its capacity to produce accurate completions for a specific task.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215573767_45935edce0_o.png" >
  <br>
  <i>LLMs fine-tuning at a high level</i>
</p>

### **4.2. Instruction Tuning**

#### **4.2.1. What is Instruction Tuning?**

One effective strategy, referred to as `instruction fine-tuning`, proves to be highly beneficial for enhancing a model's performance across various tasks. `This fine-tuning method trains the model using examples that illustrate how it should respond to specific instructions`.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53233362960_1b686b8dab_o.png" >
  <br>
  <i>Using prompts to fine-tune LLMs with instruction</i>
</p>

In both of these examples, the instruction is clear: "classify this review." The expected completion is a text string that begins with "sentiment" and is followed by either "positive" or "negative." Our training dataset consists of numerous pairs of prompt-completion examples tailored to the specific task we're focusing on, and each of these pairs includes a well-defined instruction.

For example, if our aim is to fine-tune our model and enhance its summarization capabilities, we would assemble a dataset comprising examples that commence with an explicit instruction like "summarize," followed by the subsequent text or a similar phrase. On the other hand, if our focus is on refining the model's translation skills, our dataset would encompass instructions such as "translate this sentence." These prompt-completion examples serve as a training ground, enabling the model to acquire the ability to generate responses that align with the provided instructions.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216840104_7c02740887_o.png" >
  <br>
  <i>Instruction fine-tuning</i>
</p>

Instruction fine-tuning, where all of the model's weights are updated is known as full fine-tuning. The process results in a new version of the model with updated weights. It is important to note that just like pre-training, full fine tuning requires enough memory and compute budget to store and process all the gradients, optimizers and other components that are being updated during training.

#### **4.2.2. Prepare training data**

The first step is to prepare our training data. While there are many publicly available datasets that have been used for training earlier language models, most of them lack the specific formatting required for instruction-based tasks.

Luckily, developers have assembled prompt template libraries that can be used to take existing datasets. These libraries enable the transformation of existing datasets, such as the extensive Amazon product reviews dataset, into instruction prompt datasets suitable for fine-tuning. Prompt template libraries offer a variety of templates designed for different tasks and datasets. As an example, here are three prompts tailored for use with the Amazon reviews dataset, serving purposes like model fine-tuning for classification, text generation, and text summarization tasks.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216451661_b978510587_o.png" >
  <br>
  <i>Sample prompt instruction templates</i>
</p>

In each case, we take the original review, labeled as `review_body`, and combine it with a template. The template includes an instruction, such as "predict the associated rating", "generate a star review", or "provide a short sentence describing the following product review". The result is a prompt that now contains both an instruction and the example from the data set.

Once we have our instruction-based dataset prepared, much like in standard supervised learning, we devide the dataset into training, validation, and test sets for further model development and evaluation.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216840099_ea58bbf788_o.png" >
  <br>
  <i>Split prepared instruction data into train/val/test</i>
</p>

#### **4.2.3. Instruction tuning process**

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216451651_688d9dcc8a_o.png" >
  <br>
  <i>Overall process of Instruction tuning LLMs</i>
</p>

During the fine-tuning process, we select prompts from our training dataset and feed them to the Large Language Model (LLM), which generates completions. Next, we compare the LLM's completion with the expected response specified in the training data.

In this example, the model didn't do a great job, it classified the review as neutral, but it is clearly very positive. Remember that the output of an LLM is a probability distribution across tokens. So we can compare the distribution of the completion and that of the training label and use the standard crossentropy function to calculate loss between the two token distributions. This calculated loss is then used to update the model weights through standard backpropagation. This process is repeated for numerous batches of prompt-completion pairs over several training epochs to improve the model's performance on the task.

Similar to standard supervised learning, we can establish distinct evaluation steps to assess our LLM performance using the holdout validation dataset. This will provide us with the validation accuracy. After completing the fine-tuning process, we can conduct a final performance evaluation using the holdout test dataset, which will yield the test accuracy.

The fine-tuning process results in an enhanced version of the base model, often referred to as an `instruction model`, which is more proficient at the tasks we are concerned with. Fine-tuning with instruction prompts has become the prevailing approach for enhancing LLMs. Therefore, when we encounter the term "fine-tuning" in this context, it commonly refers to instruction fine-tuning.

#### **4.2.4. Instruction fine-tuning for a single task**

While LLMs can perform many different language tasks within a single model, however, in some applications, we might require the model to excel in just one specific task. In this case, we can fine-tune a pre-trained model to improve performance on only the task that we need.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217005155_4fb40f6475_o.png" >
  <br>
  <i>Instruction fine-tuning for a single task</i>
</p>

For example, consider the task of summarization, for which we have a dataset of examples. Surprisingly, it's possible to achieve excellent results with relatively few examples (just 500 to 1,000 examples can lead to commendable performance), which contrast to the billions of pieces of text that the model encountered during its pre-training phase

However, there is a potential drawback to fine-tuning on a single task. The process may lead to a phenomenon called `catastrophic forgetting`. While it can significantly improve performance on the specific fine-tuning task, it may degrade performance on other tasks. This happens because the full fine-tuning process modifies the weights of the original LLM.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216887474_e4db33a37e_o.png" >
  <img src="https://live.staticflickr.com/65535/53215621407_717a764aaf_o.png" >
  <br>
  <i>Catastrophic forgetting example</i>
</p>

For the above example, while fine-tuning can improve the ability of a model to perform sentiment analysis on a review and result in a quality completion, the model may forget how to do other tasks.

To mitigate catastrophic forgetting, the first step is to determine whether it genuinely affects our specific use case:

- If all we need is reliable performance on the single task we fine-tuned on, it may not be an issue that the model can't generalize to other tasks.

- If we need the model to maintain its multitask generalized capabilities, we can perform fine-tuning on multiple tasks at one time. Good multitask fine-tuning may require 50-100,000 examples across many tasks, and so will require more data and compute to train.

- Our second option is to perform parameter `efficient fine-tuning (PEFT)` instead of full fine-tuning. PEFT shows greater robustness to catastrophic forgetting since this technique preserves the weights of the original LLM and trains only a small number of task-specific adapter layers and parameters.

#### **4.2.5. Instruction fine-tuning for multiple tasks**

Multitask fine-tuning is an extension of single task fine-tuning, where the training dataset is comprised of example inputs and outputs for multiple tasks.

Here, the dataset contains examples that instruct the model to perform various tasks, including summarization, review rating,... The model is trained on this diverse dataset, allowing it to enhance its performance across all these tasks simultaneously. This mitigates the issue of catastrophic forgetting. Over numerous training epochs, the calculated losses across these diverse examples are used to update the model's weights. The result is an instruction-tuned model that becomes proficient at handling a wide array of tasks simultaneously.

One drawback to multitask fine-tuning is that it requires a lot of data. We may need as many as 50,000 to 100,000 examples in our training set. However, the effort put into assembling this data is often well worth it, as the resulting models are often very capable to obtain good performance at many tasks.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216513706_98a01f27af_o.png" >
  <br>
  <i>Instruction fine-tuning for a multiple tasks</i>
</p>

### **4.2. Parameter efficient fine-tuning (PEFT)**

#### **4.2.1. PEFT vs Full Fine-tuning**

Training LLMs is computationally intensive. Full fine-tuning demands significant computational resources, not only for storing the model but also for various other parameters essential during the training process.

Even if our computer can accommodate the model weights, which can be on the order of hundreds of gigabytes for the largest models, we must also have the ability to allocate memory for optimizer states, gradients, forward activations, and temporary memory throughout the training process. These additional components can be many times larger than the model itself and can rapidly become too large to be effectively handled on consumer-grade hardware.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216863203_af42ce91ec_o.png" >
  <br>
  <i>Full fine-tuning problems</i>
</p>

Unlike full fine-tuning, where every model weight is adjusted in supervised learning, Parameter-Efficient Fine-Tuning (PEFT) methods involve updating only a limited portion of the model's parameters. Some approaches, known as `path techniques`, keep the majority of the model weights unchanged and concentrate on fine-tuning specific model parameters, such as particular layers or components. Other techniques don't modify the original model weights at all; instead, they introduce a small number of new parameters or layers and fine-tune only these newly added components.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216551471_e05a85364a_o.png" >
  <img src="https://live.staticflickr.com/65535/53215675922_52dd0ab86f_o.png" >
  <br>
  <i>PEFT methods is more efficient than full fine-tuning</i>
</p>

With PEFT, most or all of the LLM weights are kept frozen. As a result, the number of trained parameters is much smaller than the number of parameters in the original LLM. In some cases, just 15-20% of the original LLM weights. This makes the memory requirements for training much more manageable. And because the original LLM is only slightly modified or left unchanged, PEFT is less prone to the catastrophic forgetting problems of full fine-tuning.

Full fine-tuning generates a new model version for each task that is trained. Each of these is the same size as the original model, potentially leading to storage challenges when fine-tuning for multiple tasks. With PEFT, we train only a small number of weights, which results in a much smaller footprint overall,  which can be as small as megabytes, depending on the specific task.

The newly trained parameters are combined with the original LLM weights during inference. These PEFT weights are trained individually for each task and can be easily swapped out for inference, allowing for efficient adaptation of the original model to multiple tasks.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216940849_49a458fdb6_o.png" >
  <img src="https://live.staticflickr.com/65535/53216863208_6d72bc0dc9_o.png" >
  <br>
  <i>PEFT saves space and is more flexible than full fine-tuning</i>
</p>

There are several methods we can use for PEFT, each with trade-offs on parameter efficiency, memory efficiency, training speed, model quality, and inference costs. There are three main classes of PEFT methods.

- `Selective methods` are those that fine-tune only a subset of the original LLM parameters. There are several approaches that we can take to identify which parameters we want to update such as training specific components of the model, particular layers, or individual parameter types. The performance of these methods varies, and there are trade-offs between parameter efficiency and computational efficiency, which we won't delve into further in this report.

- `Reparameterization methods` also work with the original LLM parameters, but reduce the number of parameters to train by creating new low rank transformations of the original network weights. A commonly used technique of this type is `LoRA`.

- `Additive methods` execute fine-tuning while keeping all the original LLM weights frozen and introducing new trainable components. Within this category, there are two main approaches:
  - `Adapter methods` add new trainable layers to the architecture of the model, typically inside the encoder or decoder components after the attention or feed-forward layers.
  - `Soft prompt methods`, on the other hand, keep the model architecture unchanged, and focus on manipulating the input to enhance performance. This can be done by adding trainable parameters to the prompt embeddings or keeping the input fixed and retraining the embedding weights. In this report, we'll take a look at a specific soft prompts technique called `prompt tuning`.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216908848_4e35e9e5c7_o.png" >
  <br>
  <i>PEFT methods</i>
</p>

#### **4.2.1. PEFT techniques 1: LORA**

`Low-rank Adaptation`, or `LoRA` for short, is a PEFT technique that falls into the reparameterization category.

In LoRA, the idea is to make fine-tuning more efficient by reducing the number of parameters that need to be adjusted. To do this, we start with the original model and freeze all of its parameters. Then, we introduce two smaller matrices that represent rank decomposition alongside the original weights. These smaller matrices are designed to have dimensions that, when multiplied together, result in a matrix with the same size as the weights they are modifying. While we keep the original weights unchanged and frozen, we train these smaller matrices using standard supervised learning methods.

For inference, the two low-rank matrices are multiplied together to create a matrix with the same dimensions as the frozen weights. Then, we add this to the original weights and replace them in the model with these updated values. we now have a LoRA fine-tuned model that can carry out our specific task. Because this model has the same number of parameters as the original, there is little to no impact on inference latency.

When it comes to making predictions (inference), we multiply the two low-rank matrices together to create a matrix with the same dimensions as the frozen weights. Then, we add this matrix the original weights and replace them in the model with these updated values. This results in a LoRA fine-tuned model that is capable of performing the desired task. The benefit is that this modified model still has the same number of parameters as the original one, so it doesn't significantly impact the time it takes to make predictions (inference latency).

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216598686_4a2962010e_o.png" >
  <br>
  <i>LoRA Process</i>
</p>

Researchers have found that applying LoRA to just the self-attention layers of the model is often enough to fine-tune for a task and achieve better performance. We can also use LoRA on other components like the feed-forward layers., but most of the parameters of LLMs are in the attention layers, we get the biggest savings in trainable parameters by applying LoRA to these attention layer weight matrices.

For an example that used in the transformer architecture described in the "Attention is All You Need" paper. By updating the weights of these new low-rank matrices instead of the original weights, we can reduce 86% parameters in training.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216912613_57662f24a4_o.png" >
  <br>
  <i>LoRA Efficient example</i>
</p>

Because the rank-decomposition matrices in LoRA are small, we have the flexibility to fine-tune a different set for each specific task and switch them during inference by updating the model's weights. Here's how it works:

Suppose we've trained a set of LoRA matrices for a particular task, let's call it Task A. To perform inference for this task, we multiply these matrices and combine the result with the original frozen weights. This new set of weights, formed by adding these matrices to the originals, replaces the original weights in our model. This updated model can then be used for making predictions related to Task A.

If our objective switches to a different task, say Task B, we don't need to start from scratch. Instead, we use the LoRA matrices we previously trained for Task B, multiply them together, add the result to the original weights, and once again update the model. This enables us to efficiently adapt the model for different tasks without retraining the entire network.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217105405_e9c0df6177_o.png" >
  <br>
  <i>LoRA Adaptable for multiple tasks</i>
</p>

Reseacher has clarify that LoRA fine-tuning is not always as effective as full fine-tuning. In some cases, it can lead to a loss in performance. However, it is a good trade-off when we need to fine-tune a model for a task and we have limited compute resources. Moreover, LoRA is a good option when we require quick adaptability across multiple tasks.

LoRA rank is a hyperparameter that we can tune. In general, the higher the rank, the more parameters we have to train, and the more compute we need. However, higher rank matrices can capture more information from the original weights and so can lead to better performance. But there is a point that the performance gain is not worth the additional compute cost. So we should experiment with different rank values to find the best trade-off between performance and compute efficiency.

#### **4.2.2. PEFT techniques 2: Soft Prompt Tuning**

`Prompt tuning` might sound similar to prompt engineering, but these two concepts are distinct.

In prompt engineering, the focus is on adjusting the language or structure of the prompt to achieve the desired output. This can involve making simple changes, such as using different words or phrases, providing examples for one or few-shot inference. The aim is to help model understand the nature of the task and improving its generated output.

With prompt tuning, we add additional trainable tokens to our prompt and leave it up to the supervised learning process to determine their optimal values.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217013089_278acb5dbd_o.png" >
  <br>
  <i>Prompt tuning adds trainable “soft prompt” to inputs</i>
</p>

A `soft prompt` is a collection of trainable tokens that is added to the beginning of the embedding vectors representing input text. These soft prompt vectors have the same length as the embedding vectors for the language tokens. Typically, having somewhere between 20 and 100 tokens in the soft prompt is enough to achieve good performance. Through supervised learning, the model learns the optimal values for these virtual tokens to maximize its performance on a given task.

In prompt tuning, the weights of model remain fixed, and the model itself is not updated. Instead, the soft prompt's embedding vectors evolve over time to improve the model's ability to complete the prompt. Prompt tuning is an efficient strategy in terms of parameters because it involves training only a small number of parameters, in contrast to the millions or even billions of parameters in full fine-tuning.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216933748_0cf6db987c_o.png" >
  <br>
  <i>Full Fine-tuning vs prompt tuning</i>
</p>

we can train a different set of soft prompts for each task and then easily swap them out at inference time. we can train a set of soft prompts for one task and a different set for another. To use them for inference, we prepend our input prompt with the learned tokens to switch to another task, we simply change the soft prompt. Soft prompts are very small on disk, so this kind of fine tuning is extremely efficient and flexible. we'll notice the same LLM is used for all tasks, all we have to do is switch out the soft prompts at inference time.

With the soft prompt approach, we can train distinct sets of soft prompts for each task and switch between them during inference. To use them during inference, we simply add the learned soft prompt tokens to the beginning of our input prompt. When we want to switch to a different task, all we have to do is change the soft prompt we prepend to the input.

An advantage is that soft prompts are very small on disk, so this kind of fine tuning is extremely efficient and flexible.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216623831_3bb980dd6a_o.png" >
  <br>
  <i>Soft prompt tuning for multiple task</i>
</p>

## **5. Improve performance and alignment of LLMs using Reinforcement Learning From Human Feedback (RLHF)**

### **5.1. Aligning LLMs with human preferences**

The goal of fine-tuning with instructions is to further train our models for better understand prompts and generate more human-like responses. This process can significantly boost a model's performance compared to its original pre-trained state and result in more natural and human-sounding language generation.

However, we have a new set of challenges where LLMs behaving badly in use such as using toxic language in their completions, replying in combative and aggressive voices, and providing detailed information about dangerous topics. These problems exist because large models are trained on vast amounts of texts data from the Internet where such language appears frequently.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215878552_f98048b7ce_o.png" >
  <br>
  <i>Examples of LLMs behaving badly</i>
</p>

We can address this problem by additional fine-tuning with human feedback to make model better align with human preferences and increase the helpfulness, honesty, and harmlessness of the completions. This further training can also help to decrease the toxicity, often models responses and reduce the generation of incorrect information.

### **5.2. How RLHF works to improve alignment of LLMs**

Reinforcement learning from human feedback (RLHF) is a technique to fine-tune LLMs with human feedback. RHLF can help to produce better responses than a pretrained model, an instruct fine-tuned model, and even the reference human baseline.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217314915_6036c8b267_o.png" >
  <br>
  <i>Result of fine-tuning with human feedback</i>
</p>

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217196424_9231d7d97b_o.png" >
  <br>
  <i>Objective of RLHF</i>
</p>

As the name suggests, RLHF uses reinforcement learning to finetune the LLMs with human feedback data, resulting in a model that is better aligned with human preferences. By this way, RLHF can help minimize the potential for harm, maximize usefulness and relevance of LLMs. Now, how RL is extended to the case of fine-tuning LLMs with RLHF?

- Here, the agent's policy that guides the actions is the LLM, and its objective is to generate text more aligned with human preferences, for example, helpful, accurate, and non-toxic.

- The environment is the context window of the model where text can be entered via a prompt. The state that the model considers before taking an action is the current context. That means any text currently contained in the context window.

- The action is the act of generating text. This could be a single word, a sentence,... depending on the task specified by the user. The action space is the token vocabulary, meaning all the possible tokens that the model can choose from to generate the completion. At any given moment, the action that the model will take, meaning which token it will choose next, depends on the prompt text in the context and the probability distribution over the vocabulary space.

- The reward is assigned based on how closely the completions align with human preferences. However, determining these rewards can be a complex task since human responses to language can vary significantly. One approach to tackle this is to have a human evaluate all of the model's completions using an alignment metric, like whether the generated text is toxic or non-toxic. The feedback from this evaluation can be expressed as a scalar value, such as 0 (poor alignment with preferences) or 1 (strong alignment with preferences).

- The LLM weights are then updated iteratively to maximize the reward obtained from the human classifier, enabling the model to generate non-toxic completions.

However, obtaining human feedback can be time consuming and expensive. To make it more practical and scalable, we can use an additional model, known as the `reward model`, to classify the outputs of the LLM and evaluate the degree of alignment with human preferences

We'll start with a small number of human examples to train the `reward model` by traditional supervised learning methods. Later, reward model is used to assess the output of the LLM and assign a reward value, which is then used to adjust the weights of the LLM and train a new version that is more aligned with human preferences.

The reward model is the central component of the reinforcement learning process. It encodes all of the preferences learned from human feedback, and plays a central role in guiding how the model's weights are updated over multiple iterations to generate more desirable responses.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215932087_1491c0e98a_b.jpg" >
  <br>
  <i>Flow of fine-tuning LLMs with RLHF</i>
</p>

### **5.4. Obtaining human feedback**

The initial step in fine-tuning Large Language Models (LLMs) with Reinforcement Learning from Human Feedback (RLHF) involves selecting an appropriate LLMs model and creating a dataset for human feedback. The chosen model should have capability to perform the task at hand like text summarization, question answering,... Typically, starting with a pretrained model that has been fine-tuned across various tasks and has a broad range of capabilities is a sensible approach.

With this selected LLM, you'll use a prompt dataset to generate multiple responses for each prompt. The prompt dataset consists of numerous prompts, and each of these prompts is processed by the LLM to produce a set of potential completions. This dataset serves as the foundation for collecting human feedback and further refining the model's performance.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217136603_fa7cdcdcfd_o.png" >
  <br>
  <i>Prepare dataset for human feedback</i>
</p>

The next step is to collect feedback from human labelers on the completions generated by the LLM. First, we must decide what criterion we want human labelers to assess the completions on. This could be any of the issues like helpfulness or toxicity. Once we've decided, we will then ask the labelers to assess each completion in the data set based on that criterion.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217217294_d7ca21ddbc_b.jpg" >
  <br>
  <i>Sample instructions for human labelers</i>
</p>

For the below example, we pass a prompt to the LLM, which then generates 3 different completions. The task for labelers is to rank the 3 completions from the most helpful to least helpful. This process then gets repeated for many prompt completion sets, building up a data set that can be used to train the reward model that will be able to carry out this work instead of the humans.

The same prompt completion sets are usually assigned to multiple human labelers to establish consensus and minimize the impact of poor labelers in the group. Like the third labeler here, whose responses disagree with the others and may indicate that they misunderstood the instructions. The clarity of our instructions to human lablers can make a big difference on the quality of the human feedback we obtain.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217217284_a476afdd15_o.png" >
  <br>
  <i>Example of collecting human feedback</i>
</p>

Once your human labelers have completed their assessments off the prompt completion sets, you have all the data you need to train the reward model which will be used instead of humans to classify model completions during the reinforcement learning fine-tuning process.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216828061_03576f4f7a_o.png" >
  <br>
  <i>Ranking data into a pairwise comparison</i>
</p>

Before start training the reward model, we need to convert the ranking data into a pairwise comparison of completions. In other words, all possible pairs of completions from the available choices to a prompt should be classified as 0 or 1 score.

For instance, if you have three completions for a prompt and human labelers have ranked them as 2, 1, 3, where 1 is the highest rank for the most preferred response, you'll generate three pairs. Depending on the number N of alternative completions per prompt, you will have N choose 2 combinations ($`C_{N}^{2}`$).

For each pair, you will assign a reward of 1 for the preferred response and a reward of 0 for the less preferred response. Then you'll reorder the prompts so that the preferred option comes first. This is an important step because the reward model expects the preferred completion to come first.

Once you've completed this data restructuring, the human responses will be in the correct format for training the reward model. Note that while like or dislike feedback is often easier to gather than ranking feedback, ranked feedback gives more prompt completion data to train your reward model. As you can see, here you get three prompt completion pairs from each.

### **5.5. Training the reward model**

Once we've gathered the necessary data, you can proceed to train the reward model. As you complete the training of this reward model, you'll no longer require human intervention in the process. Instead, the reward model takes on the role of humans and automatically selects the preferred completion during the feedback process.

This reward model is usually also a language model. For example, a BERT is trained using supervised learning methods on the pairwise comparison data that you prepared from the human labeler's assessment of the prompts. For a given prompt X, the reward model learns to favor the human-preferred completion $`y_j`$, while minimizing the log of sigmoid of the reward difference $`r_j - r_k`$

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217237519_cb88078fd3_o.png" >
  <br>
  <i>Training the reward model</i>
</p>

Once the reward model has been trained on the human rank prompt-completion pairs, we can use it as a binary classifier to provide a set of logits across the positive and negative classes. Logits are the unnormalized model outputs before applying any activation function. If you apply a Softmax function to the logits, you will get the probabilities

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215973357_b8812817ec_o.png" >
  <br>
  <i>Use the reward model</i>
</p>

When the goal is to detoxify Large Language Models (LLMs), the reward model's purpose is to determine whether a completion contains hate speech. In this context, you're optimizing for the positive class, which is "not hate" and trying to avoid the negative class, which is "hate."

The highest value associated with the positive class, which represents "not hate," is the reward value used in Reinforcement Learning from Human Feedback (RLHF). In RLHF, a good reward is assigned for non-toxic completions, as demonstrated in the first example, while a poor reward is given for toxic completions, as illustrated in the second example. This process helps train the model to favor responses that are free of hate speech and avoid generating toxic content.

### **5.6. Fine-tuning with RLHF**

Here, we start with a model that already has good performance on your task of interests. We'll work to align an instruction fine-tuned LLM.

- First, we'll pass a prompt from our prompt dataset. In this case, "a dog is", to the instruct LLM, which then generates a completion, in this case "a furry animal".

- Next, we sent this completion, and the original prompt to the reward model as the prompt-completion pair. The reward model evaluates the pair and returns a reward value.

- Then pass this reward value to the reinforcement learning algorithm to update the weights of the LLM.

These series of steps together forms a single iteration of the RLHF process. We will continue this iterative process until our model is aligned based on some evaluation or stopping criteria.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217254279_f5e7dd5f10_o.png" >
  <img src="https://live.staticflickr.com/65535/53217173803_17b7c76cc4_o.png" >
  <img src="https://live.staticflickr.com/65535/53216864216_0082892685_o.png" >
  <img src="https://live.staticflickr.com/65535/53217173798_fb4367b2eb_o.png" >
  <br>
  <i>Fine-tuning with RLHF</i>
</p>

There are also several different algorithms that you can use for this part of the RLHF process. A popular choice is proximal policy optimization or PPO for short.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53227858261_78c2ea0097_b.jpg" >
  <br>
  <i>PPO</i>
</p>

### **5.7. Proximal policy optimization (PPO)**

Proximal policy optimization (PPO) is a powerful algorithm for solving reinforcement learning problems that is used to optimize the policy of an agent, in this case the LLM to be more aligned with human preferences. We obtain agent's training stability by avoiding too large policy update. The updates are small and within a bounded region, resulting in an updated LLM that is close to the previous version, hence the name Proximal Policy Optimization.

Using PPO in aligning LLMs has 2 main phases.

- Phase 1: Create completions, calculate rewards and value losses.
- Phase 2: Update model with an objective function.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53226984382_d393ec718d_b.jpg" >
  <br>
  <i>PPO phases</i>
</p>

#### **5.7.1. Phase 1**

In Phase I, the LLM, is used to carry out a number of experiments, completing the given prompts. These experiments allow you to update the LLM against the reward model in Phase II. Remember that the reward model captures the human preferences. For example, the reward can define how helpful, harmless, and honest the responses are. The expected reward of a completion is an important quantity used in the PPO objective. We estimate this quantity through a separate head of the LLM called the value function.

Assume a number of prompts are given. First, LLMs generate responses to the prompts, then we calculate the reward for the prompt completions using the reward model. Then, the value function estimates the expected total reward for a given state s. In other words, as the LLM generates each token of a completion, you want to estimate the total future reward based on the current sequence of tokens.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53228235899_d316212798_b.jpg" >
  <img src="https://live.staticflickr.com/65535/53228166348_0c3f501c15_o.png" >
  <br>
  <i>Create completions and calculate rewards</i>
</p>

The goal is to minimize the value loss that is the difference between the actual future total reward and its approximation to the value function. The value loss makes estimates for future rewards more accurate. The value function is then used in Advantage Estimation in Phase 2.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53228166353_734e5ccc2b_b.jpg" >
  <br>
  <i>Calculate value loss</i>
</p>

#### **5.7.2. Phase 2**

In Phase 2, you make a small updates to the model and evaluate the impact of those updates on your alignment goal for the model. The model weights updates are guided by the prompt completion, losses, and rewards. PPO also ensures to keep the model updates within a certain small region called the `trust region`. Ideally, this series of small updates will move the model towards higher rewards.

##### **The intuition behind PPO**

The idea with Proximal Policy Optimization (PPO) is that we want to improve the training stability of the policy by limiting the change you make to the policy at each training epoch: we want to avoid having too large policy updates for two reasons:

- We know empirically that smaller policy updates during training are more likely to converge to an optimal solution.
- A too big step in a policy update can result in getting a bad policy and having a long time or even no possibility to recover.

<p align="center">
  <img src="https://huggingface.co/blog/assets/93_deep_rl_ppo/cliff.jpg" >
  <br>
  <i>Taking smaller policy updates improve the training stability</i>
</p>

So with PPO, we update the policy conservatively. To do so, we need to measure how much the current policy changed compared to the former one using a ratio calculation between the current and former policy. And we clip this ratio in a range $`[1-\epsilon, 1+\epsilon]`$ where $`\epsilon`$ is a hyperparameter. That mean we remove the incentive for the current policy to go too far from the old one.

##### **The Policy Objective Function in RL**

<p align="center">
  <img src="https://huggingface.co/blog/assets/93_deep_rl_ppo/lpg.jpg" >
  <br>
  <i>The Policy Objective Function</i>
</p>

The idea of the policy objective function in RL was that by taking a gradient ascent step on this function (equivalent to taking gradient descent of the negative of this function), we would push our agent to take actions that lead to higher rewards and avoid harmful actions. However, the problem comes from the step size:

- Too small, the training process was too slow
- Too high, there was too much variability in the training

Here with PPO, the idea is to constrain our policy update with a new objective function called the Clipped surrogate objective function that will constrain the policy change in a small range using a clip. This new function is designed to avoid destructive large weights updates :

<p align="center">
  <img src="https://huggingface.co/blog/assets/93_deep_rl_ppo/ppo-surrogate.jpg" >
  <br>
  <i>PPO's Clipped surrogate objective function</i>
</p>

##### **The Ratio Function**

<p align="center">
  <img src="https://huggingface.co/blog/assets/93_deep_rl_ppo/ratio1.jpg" >
  <img src="https://huggingface.co/blog/assets/93_deep_rl_ppo/ratio2.jpg" >
  <br>
  <i>The Ratio Function</i>
</p>

It’s the probability of taking action $`a_t`$ at state $`s_t`$​ in the current policy divided by the previous one. As we can see, $`r_{t}(\theta)`$ denotes the probability ratio between the current and old policy:

- If $`r_{t}(\theta) > 1`$, the action $`a_t`$ at state $`s_t`$​ is more likely to be taken in the current policy than in the previous one.
- If $`0 < r_{t}(\theta) < 1`$, the action $`a_t`$ at state $`s_t`$​ is less likely to be taken in the current policy than in the old one.

So this probability ratio is an easy way to estimate the divergence between old and current policy.

##### **The unclipped part of the Clipped Surrogate Objective function**

<p align="center">
  <img src="https://huggingface.co/blog/assets/93_deep_rl_ppo/unclipped1.jpg" >
  <br>
  <i>The unclipped part</i>
</p>

This ratio can replace the log probability we use in the policy objective function. This gives us the left part of the new objective function: multiplying the ratio by the advantage. However, without a constraint, if the action taken is much more probable in our current policy than in our former, this would lead to a significant policy gradient step and, therefore, an excessive policy update.

##### **The clipped Part of the Clipped Surrogate Objective function**

<p align="center">
  <img src="https://huggingface.co/blog/assets/93_deep_rl_ppo/clipped.jpg" >
  <br>
  <i>The clipped part</i>
</p>

Consequently, we need to constrain this objective function by penalizing changes that lead to a ratio away from 1 (in the paper, the ratio can only vary from 0.8 to 1.2). By clipping the ratio, we ensure that we do not have a too large policy update because the current policy can't be too different from the older one.

To do that, we have two solutions:

- TRPO (Trust Region Policy Optimization) uses KL divergence constraints outside the objective function to constrain the policy update. But this method is complicated to implement and takes more computation time.
- PPO clip probability ratio directly in the objective function with its Clipped surrogate objective function.

This clipped part is a version where $`r_{t}(\theta)`$ is clipped between $`[1-\epsilon, 1+\epsilon]`$.

With the Clipped Surrogate Objective function, we have two probability ratios, one non-clipped and one clipped in a range. Epsilon is a hyperparameter that helps us to define this clip range and is set to 0.2 in the paper.

Then, we take the minimum of the clipped and non-clipped objective, so the final objective is a lower bound (pessimistic bound) of the unclipped objective. Taking the minimum of the clipped and non-clipped objective means we'll select either the clipped or the non-clipped objective based on the ratio and advantage situation.

##### **Visualize the Clipped Surrogate Objective function**

<p align="center">
  <img src="https://huggingface.co/blog/assets/93_deep_rl_ppo/recap.jpg" >
  <br>
  <i>6 different situations of the Clipped Surrogate Objective function</i>
</p>

`Case 1 and 2:` the ratio is between the range $`[1-\epsilon, 1+\epsilon]`$ so he clipping does not apply.

- In situation 1, we have a positive advantage: the action is better than the average of all the actions in that state. Therefore, we should encourage our current policy to increase the probability of taking that action in that state. Since the ratio is between intervals, we can increase our policy's probability of taking that action at that state.

- In situation 2, we have a negative advantage: the action is worse than the average of all actions at that state. Therefore, we should discourage our current policy from taking that action in that state. Since the ratio is between intervals, we can decrease the probability that our policy takes that action at that state.

`Case 3 and 4:` the ratio is below the range.

- If the probability ratio is lower than $`[1-\epsilon]`$, the probability of taking that action at that state is much lower than with the old policy.
- If, like in situation 3, the advantage estimate is positive (A>0), then you want to increase the probability of taking that action at that state.
- But if, like situation 4, the advantage estimate is negative, we don't want to decrease further the probability of taking that action at that state. Therefore, the gradient is = 0 (since we're on a flat line), so we don't update our weights.

`Case 5 and 6:` the ratio is above the range.

- If the probability ratio is higher than $`[1+\epsilon]`$, the probability of taking that action at that state in the current policy is much higher than in the former policy.
- If, like in situation 5, the advantage is positive, we don't want to get too greedy. We already have a higher probability of taking that action at that state than the former policy. Therefore, the gradient is = 0 (since we're on a flat line), so we don't update our weights.
- If, like in situation 6, the advantage is negative, we want to decrease the probability of taking that action at that state.

`So to sum up,` we update the policy with the unclipped objective part only if:

- Our ratio is in the range $`[1-\epsilon, 1+\epsilon]`$.
- Our ratio is outside the range, but the advantage leads to getting closer to the range.
  - Being below the ratio but the advantage is > 0
  - Being above the ratio but the advantage is < 0

When the minimum is the clipped objective part, we don't update our policy weights since the gradient will equal 0 (the derivative of both $`(1-\epsilon)*A`$ and $`(1+\epsilon)*A`$ is 0).

##### **Calculate the entropy loss**

We also have an additional component which is entropy loss. While the policy loss moves the model towards alignment goal, entropy allows the model to maintain creativity. If you kept entropy low, you might end up always completing the prompt in the same way as shown here. Higher entropy guides the LLM towards more creativity. This is similar to the temperature setting of LLM. The difference is that the temperature influences model at the inference time, while the entropy influences during training.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53226984407_bf878c1b49_b.jpg" >
  <br>
  <i>Entropy loss</i>
</p>

##### **The final objective function of PPO**

The final Clipped Surrogate Objective Loss for PPO is a combination of Clipped Surrogate Objective function, Value Loss Function and Entropy bonus. The C1 and C2 coefficients are hyperparameters.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53228166383_7a1dab6d16_b.jpg" >
  <br>
  <i>The final objective function</i>
</p>

The PPO objective updates the model weights through back propagation over several steps which updates the model towards human preference in a stable manner. Once the model weights are updated, PPO starts a new cycle. After many iterations, you arrive at the human-aligned LLM.

## **6. Evaluating LLMs**

### **6.1. BLEU (Bilingual Evaluation Understudy)**

`BLEU (BiLingual Evaluation Understudy)` is a metric for evaluating machine-translated text. The BLEU score is between zero and one that measures the similarity of the machine-translated text to a set of high quality reference translations. The following table shows the interpretation of BLEU scores:

| BLEU Score  | Interpretation                                            |
|-------------|-----------------------------------------------------------|
| < 0.1       | Almost useless                                            |
| 0.1 - 0.19  | Hard to get the gist                                      |
| 0.2 - 0.29  | The gist is clear, but has significant grammatical errors |
| 0.3 - 0.4   | Understandable to good translations                       |
| 0.4 - 0.5   | High quality translations                                 |
| 0.5 - 0.6   | Very high quality, adequate, and fluent translations      |
| > 0.6       | Quality often better than human                           |

#### **Precision**

Precision measures the number of words in the Predicted Sentence that also occur in the Target Sentence. We would normally compute the Precision using the formula:

```math
Precision = \frac{\text{Number of words in Predicted Sentence that also occur in Target Sentence}}{\text{Number of words in Predicted Sentence}}
```

For example:

- Target Sentence: He eats an apple
- Predicted Sentence: He ate an apple

So the precision for the above example is 3/4. But using Precision like this is not good enough. There are two cases that we still need to handle.

- The first issue is that this formula allows us to cheat:
  - Target Sentence: He eats an apple
  - Predicted Sentence: He He He
  - The above get perfect precison = 3/3 = 1
- Secondly, there are many correct ways to express the same sentence. In many NLP models, we might be given multiple acceptable target sentences that capture these different variations.

We need to modify the formula to handle these two cases. We account for these two scenarios using a modified Precision formula which we’ll call `“Clipped Precision”`. Let’s say, that we have the following sentences:

- Target Sentence 1: He eats a sweet apple
- Target Sentence 2: He is eating a tasty apple
- Predicted Sentence: He He He eats tasty fruit

We now do two things differently:

- We compare each word from the predicted sentence with all of the target sentences. If the word matches any target sentence, it is considered to be correct.
- We limit the count for each correct word to the maximum number of times that that word occurs in the Target Sentence. This helps to avoid the Repetition problem. This will become clearer below.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*ccCNFbdGeGzZr-SM.png" >
  <br>
  <i>Clip precision example</i>
</p>

For instance, the word “He” occurs only once in each Target Sentence. Therefore, even though “He” occurs thrice in the Predicted Sentence, we ‘clip’ the count to one, as that is the maximum count in any Target Sentence.

```math
Clipped Precision = \frac{\text{Clipped number of correct predicted words}}{\text{ Number of total predicted words}}
```

So the clipped precision for the above example is 3/6.

#### **Calculate BLEU score**

Let’s say we have an NLP model that produces a predicted sentence as below. For simplicity, we will take just one Target Sentence, but as in the example above, the procedure for multiple Target Sentences is very similar.

- Target Sentence: The guard arrived late because it was raining
- Predicted Sentence: The guard arrived late because of the rain

The first step is to compute Precision scores for 1-grams through 4-grams. We use the Clipped Precision method that we just discussed.

```math
\text{Precision 1-gram} = \frac{\text{Number of correct predicted 1-grams}}{\text{Number of total predicted 1-grams}}
```

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/0*uInR1otrb3d3pKtV.png" >
  <br>
  <i>Precision 1-gram (p1) = 5 / 8</i>
</p>

```math
\text{Precision 2-gram} = \frac{\text{Number of correct predicted 2-grams}}{\text{Number of total predicted 2-grams}}
```

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/0*cIGCQAAdahIx_Snv.png" >
  <br>
  <i>Precision 2-gram (p2) = 4 / 7</i>
</p>

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/0*fQEPPBKQuQ0OrLNJ.png" >
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/0*JyUBZ5o2h1Z2PwXe.png" >
  <br>
  <i>Similarly, Precision 3-gram (p₃) = 3 / 6 and Precision 4-gram (p₄) = 2 / 5</i>
</p>

Next, we combine these Precision Scores using the formula below. This can be computed for different values of N and using different weight values. Typically, we use N = 4 and uniform weights $`w_n = N / 4`$

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*0Zi8SI4CkOMd7avkk9BTTA.png" >
  <br>
  <i>Precision Scores</i>
</p>

The third step is to compute a `‘Brevity Penalty’`. There are maybe some predicted sentence consisting of a single word like “The’ or “late”. For this, the 1-gram Precision would have been 1/1 = 1, indicating a perfect score. This is obviously misleading because it encourages the model to output fewer words and get a high score. To offset this, the Brevity Penalty penalizes sentences that are too short.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/0*AoJZFTk0AXqHZrM9.png" >
  <br>
  <i>Brevity Penalty</i>
</p>

Where c is predicted length (number of words in the predicted sentence) and r is target length (number of words in the target sentence). This ensures that the Brevity Penalty cannot be larger than 1, even if the predicted sentence is much longer than the target. And, if you predict very few words, this value will be small. In this example, c = 8 and r = 8, which means Brevity Penalty = 1.

Finally, to calculate the Bleu Score, we multiply the Brevity Penalty with the Geometric Average of the Precision Scores.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*gGgQvDgayGCfTkI5MFGMjg.png" >
  <br>
  <i>Bleu Score</i>
</p>

Bleu Score can be computed for different values of N. Typically, we use N = 4.

- BLEU-1 uses the unigram Precision score
- BLEU-2 uses the geometric average of unigram and bigram precision
- BLEU-3 uses the geometric average of unigram, bigram, and trigram precision and so on.

#### **Strengths of Bleu Score**

The reason that Bleu Score is so popular is that it has several strengths:

- It is quick to calculate and easy to understand.
- It corresponds with the way a human would evaluate the same text.
- Importantly, it is language-independent making it straightforward to apply to your NLP models.
- It can be used when you have more than one ground truth sentence.
- It is used very widely, which makes it easier to compare your results with other work.

#### **Weaknesses of Bleu Score**

In spite of its popularity, Bleu Score has been criticized for its weaknesses:

- It does not consider the meaning of words. It is perfectly acceptable to a human to use a different word with the same meaning eg. Use “watchman” instead of “guard”. But Bleu Score considers that an incorrect word.
- It looks only for exact word matches. Sometimes a variant of the same word can be used eg. “rain” and “raining”, but Bleu Score counts that as an error.
- It ignores the importance of words. With Bleu Score an incorrect word like “to” or “an” that is less relevant to the sentence is penalized just as heavily as a word that contributes significantly to the meaning of the sentence.
- It does not consider the order of words eg. The sentence “The guard arrived late because of the rain” and “The rain arrived late because of the guard” would get the same (unigram) Bleu Score even though the latter is quite different.

### **6.2. ROUGE (Recall-Oriented Understudy for Gissing Evaluation)**

`ROUGE` is a set of metrics used for evaluating the quality of summaries. Unlike BLEU, the ROUGE uses both recall and precision to compare model generated summaries known as candidates against a set of human generated summaries known as references.

ROUGE compares the generated summary with one or more reference summaries and calculates precision, recall, and F1-score. ROUGE scores provide insights into the summary generation capabilities of the language model. Hence, the recall, precision and F1 can be obtained as shown below:

```math
\text{Recall} = \frac{\text{Number of mathcing n-grams between candidate and reference summaries}}{\text{Number of n-grams in reference summary}}
```

```math
\text{Precision} = \frac{\text{Number of mathcing n-grams between candidate and reference summaries}}{\text{Number of n-grams in candidate summary}}
```

```math
F1 = \frac{2*Recall*Precision}{Recall + Precision}
```

Beside that, we have ROUGE-L, which is defined as the longest common subsequence (LCS) between the candidate and reference summaries. The LCS is the longest sequence of words that are common between the candidate and reference summaries. The ROUGE-L score is calculated as follows:

```math
\text{ROUGE-L Recall} = \frac{\text{LCS(candidate, reference)}}{\text{Number of words in reference summary}}
```

```math
\text{ROUGE-L Recall} = \frac{\text{LCS(candidate, reference)}}{\text{Number of words in candicate summary}}
```

```math
\text{ROUGE-L F1} = \frac{(1 + \beta^2)*Recall*Precision}{Recall + \beta^2*Precision}
```

Here, $`\beta`$ is used to balance the importance of recall and precision.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216054612_8e9fb4cbd1_o.png" >
  <img src="https://live.staticflickr.com/65535/53216054617_79f398acba_o.png" >
  <img src="https://live.staticflickr.com/65535/53217237518_726d314ee3_o.png" >
  <br>
  <i>ROUGE examples</i>
</p>

ROUGE is ussually used for evalutating text summarization tasks because it can be used to compares the generated summary with one or more reference summaries.

ROUGE has the following advantages:

- Easy to calculate and understand and is usually used to evaluate text summarization systems.
- ROUGE and its variants have worked well for summarization of single documents and short texts.

Disadvantages:

- ROUGE has the same problem as BLEU regarding the synonyms. The meaning of the n-grams is not considered and hence “fast” and “quick” will not be considered as similar.
- The score is heavily dependent on the word choice and structure of the references chosen for evaluation.

### **6.3. Perplexity**

Perplexity is a measurement that reflects `how well a model can predict the next word based on the preceding context`. The higher the likelihood the model assigns to the dataset, the lower the perplexity. `The lower the perplexity score, the better the model’s ability to predict the next word accurately.`

Mathematically, perplexity is defined as the exponential of the average negative log likelihood per token:

```math
log(Permplexity) = -\frac{1}{N}\sum_{i=1}^{N}log(P(t_i|t_1, t_2, ..., t_{i-1}))
```

Where:

- N is the number of tokens in the text corpus.
- $`log(P(t_i|t_1, t_2, ..., t_{i-1}))`$ is the predicted probability of the i-th word given the preceding words $`t_1, t_2, ..., t_{i-1}`$.

<p align="center">
  <img src="https://images.surferseo.art/42b4e02c-2bfb-4955-bd5a-f55bf90465fb.png" >
  <br>
  <i>Perplexity example</i>
</p>

Perplexity plays a crucial role in determining the success of LLMs and generative AI models for several reasons like enhance user experience, evaluation of model performance, etc. But it has some main limitations:

- Model Vocabulary Could Unfairly Influence Perplexity: Perplexity heavily relies on the model’s vocabulary and its ability to generalize unseen words. If a model comes across words or phrases that are not present in its training data, it may assign high perplexity scores even when the generated text makes sense.

- Lack of Subjectivity Consideration: Perplexity does not account for subjective factors such as style, creativity, or appropriateness in specific contexts. A model with a low perplexity may generate text that is technically accurate but lacks the desired tone, voice, or level of creativity required for certain applications.

- Contextual Understanding: Perplexity primarily focuses on the prediction of the next word based on the preceding context. However, it may not capture the model’s overall understanding of the broader context. For example, a model may assign a low perplexity score to a sentence that is technically correct but does not make sense in the given context.

- Language Ambiguity and Creativity: Perplexity does not capture the model’s ability to handle language ambiguity or generate creative and novel outputs.

- Overfitting and Generalization: Perplexity can be affected by overfitting, where a model performs exceptionally well on the training data but struggles with generalizing to unseen or real-world examples. Models that achieve low perplexity scores on a specific dataset may not perform as well on diverse inputs or in practical scenarios.

- ...

### **6.4. Benchmarking**

Beside the above evaluation methods, `benchmarking` is also a popular method for evaluating LLMs. Benchmarking is the process of comparing the performance of a model against a set of standard tasks or datasets. Benchmarking provides a standardized way to compare the performance of different models. It also helps to identify the strengths and weaknesses of a model and to understand how it performs on different tasks.

However, benchmarking has some limitations. For example, it can be difficult to design a benchmark that captures the full range of a model’s capabilities. Additionally, benchmarking may not be able to capture the nuances of a model’s performance on a specific task. Therefore, it is important to use benchmarking in conjunction with other evaluation methods.

| Framework Name | Factors Considered for Evaluation | Url Link|
| --- | --- | --- |
| Big Bench | Generalization abilities | <https://github.com/google/BIG-bench> |
| GLUE Benchmark | Grammar, Paraphrasing, Text Similarity, Inference, Textual Entailment, Resolving Pronoun References | <https://gluebenchmark.com/> |
| SuperGLUE Benchmark | Natural Language Understanding, Reasoning, Understanding complex sentences beyond training data, Coherent and Well-Formed Natural Language Generation, Dialogue with Human Beings, Common Sense Reasoning (Everyday Scenarios and Social Norms and Conventions), Information Retrieval, Reading Comprehension | <https://super.gluebenchmark.com/> |
| OpenAI Moderation API | Filter out harmful or unsafe content | <https://platform.openai.com/docs/api-reference/moderations> |
| MMLU | Language understanding across various tasks and domains | <https://github.com/hendrycks/test> |

### **6.6. Challenges with existing LLM evaluation methods**

While existing evaluation methods for Large Language Models (LLMs) provide valuable insights, they are not perfect. The common issues associated with them are:

- *Over-reliance on perplexity*: Perplexity measures how well a model predicts a given text but does not capture aspects such as coherence, relevance, or context understanding. Therefore, relying solely on perplexity may not provide a comprehensive assessment of an LLM’s quality.

- *Subjectivity in human evaluations*: Human evaluation is a valuable method for assessing LLM outputs, but it can be subjective and prone to bias. Different human evaluators may have varying opinions, and the evaluation criteria may lack consistency. Additionally, human evaluation can be time-consuming and expensive, especially for large-scale evaluations.

- *Limited reference data*: Some evaluation methods, such as BLEU or ROUGE, require reference data for comparison. However, obtaining high-quality reference data can be challenging, especially in scenarios where multiple acceptable responses exist or in open-ended tasks. Limited or biased reference data may not capture the full range of acceptable model outputs.

- *Lack of diversity metrics*: Existing evaluation methods often don’t capture the diversity and creativity of LLM outputs. That’s because metrics that only focus on accuracy and relevance overlook the importance of generating diverse and novel responses. Evaluating diversity in LLM outputs remains an ongoing research challenge.

- *Generalization to real-world scenarios*: Evaluation methods typically focus on specific benchmark datasets or tasks, which don’t fully reflect the challenges  of real-world applications. The evaluation on controlled datasets may not generalize well to diverse and dynamic contexts where LLMs are deployed.

- *Adversarial attacks*: LLMs can be susceptible to adversarial attacks such as manipulation of model predictions and data poisoning, where carefully crafted input can mislead or deceive the model. Existing evaluation methods often do not account for such attacks, and robustness evaluation remains an active area of research.

## **7. References**

[1] [Generative AI with Large Language Models Course of DeepLearning.AI on Coursera](https://www.coursera.org/learn/generative-ai-with-llms?utm_campaign=WebsiteCoursesGAIA&utm_medium=institutions&utm_source=deeplearning-ai)

[2] [Introduction to Large Language Models by Kumar Chandrakant on www.baeldung.com](https://www.baeldung.com/cs/large-language-models)

[3] [Large language model - Wikipedia](https://en.wikipedia.org/wiki/Large_language_model)
