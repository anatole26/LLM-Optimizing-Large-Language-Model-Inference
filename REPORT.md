# Transformers:

The transformer architecture was introduced by Google employees Vaswani et al. in the paper "Attention is All You Need" (2017) and revolutionized natural language processing and sequential data modelling. 

In contrast to traditional sequence models, which often lean on sequential architectures like Recurrent Neural Networks (RNNs) that process data one step at a time, transformers employ a self-attention mechanism. This mechanism enables transformers to simultaneously consider global dependencies within input sequences, irrespective of the distance between elements. Unlike RNNs, where dependencies are captured sequentially, transformers can efficiently capture intricate relationships across the entire sequence in a parallelized manner, making them highly effective for a wide range of tasks.

Here is what the transformer architecture looks like, as introduced in "Attention is All You Need" (2017):

![](https://miro.medium.com/v2/resize:fit:1400/1*BHzGVskWGS_3jEcYYi6miQ.png)

From this point on, this transformer achitecture will be sometimes refered to as the traditional transformer achitecture.

Large language models (LLM) such as LLaMA and LLaMA 2 are based on this standard transformer architecture. However, they both mildly deviate from it, and ultimately improve upon it. The main improvements leveraged by LLaMA and LLaMA 2 are as follows:

- Pre-normalization: To improve the training stability, the input of each transformer sub-layer is normalized, instead of the output being normalized. To this end, the "RMSNorm" normalizing function, introduced by Zhang and Sennrich (2019), is used. To visualize this, one can look at the schema above and picture the normalization layers (Add & Norm) before their respective attention layers (Multi-Head Attention) and feed forward layers (Feed Forward), instead of after.

- SwiGLU activation function: In the feed forward layers, the ReLU activation function is replaced by the SwiGLU activation function, introduced by Shazeer (2020). 

- Rotary Embeddings: Absolute positional embeddings that are used in the classical transformer architecture, are removed. In their place are added rotary positional embeddings (RoPE), introduced by Su et al. (2021). The use of rotary embeddings allows the attention mechanism to capture both linear and rotational positional information, providing the model with the ability to distinguish between different positions in a sequence more effectively.

- No decoder: LLaMA models are LLMs that have been trained on the next token prediction task. For predicting the next token, only self-attention layers are required. Therefore, there is only an encoder, and no decoder needed, in its transformer architecture. (Note: in the figure above, the encoder is the left part of the schema and the decoder is the right part)


In addition to these changes to the transformer architecture introduced in "Attention is All You Need" (2017), LLaMA 2 introduces a further innovation known as Grouped-Query Attention (GQA). In simple terms, GQA represents a standard practice in autoregressive decoding, involving the placement of key (K) and value (V) pairs for previous tokens in the sequence into auxiliary memory. This accelerates attention computation.

Having provided an overview of transformers and highlighted the nuanced alterations from their inception to the LLaMA and LLaMA 2 LLMs, we will now delve into a detailed examination of the key components of transformers for LLaMA models, presented in order, along with their corresponding theoretical aspects.


## LLaMA Transformer Architecture:


- **Input Embedding**: The input sequence, which is text in the context of large language models, is initially embedded into vectors, allowing the model to represent each element in a high-dimensional space. This process is essential for transforming discrete symbols into a format that can be interpretated and processed by neural networks. 

    - Tokenisation: The first step in input embedding is to break down the input sequence into smaller units, often referred to as tokens. Tokens can range from individual characters to large words. 

    - Word Embedding: Once tokenized, each token is mapped to a high-dimensional vector through a process known as word embedding. Word embeddings are pre-trained representations of words in a continuous vector space which the machines can understand and interpret. These embeddings capture the meaning and relationship between words, enabling the model to understand similarities and differences between them. 


- **Encoder**: The encoder's primary function is to process the input sequence and create a contextualized representation for each element in that sequence. An encoder is tranditionally comprised of self-attention layers and a feedforward neural network. In the case of the LLaMA framework, given the pre-normalisation of the inputs before the self-attention layers, the encoder is comprised of a Root Mean Square (RMS) Normalization layer, self-attention layers and a feedforward neural network (Note: the encoder part of the transformer archetecture is the left side of the schema above)

    - Root Mean Square (RMS) Layer Normalization of the inputs:
            
        The inputs go through a RMS function, which focuses on re-scaling invariance and regularizes the summed inputs simply according to the RMS statistic. Let $ \mathbf{X} $ be our input sequence. We have: 
        $$ \text{RMS}(\mathbf{X}) = \sqrt{\frac{1}{N}\sum_{i=1}^{N}X_i^2} $$
            
        Here, $ N $ is the length of the sequence, and $ X_i $ is the value of the i-th element. 

        Each element is then normalized in the input sequence by dividing it by the RMS value:
        $$ \text{Normalized Input}_i = \frac{{X_i}}{\text{RMS}(\mathbf{X})}g_i $$
        , where $ g $ is a learnable parameter.
        
        The traditional transformer architecture uses layer normalisation which its function works as such:
        $$ \text{LayerNorm}(\text{X}) = \frac{(\text{X} - \mu)}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta $$ 
        , where $ \text{x} $ is the input tensor, $ \mu $ is the mean of the input tensor along the normalization dimension, $ \sigma $ is the standard deviation of the input tensor along the normalization dimension, $ \epsilon $ is a small constant (usually a small positive value) added to the denominator for numerical stability, $ \gamma $ is a learnable scale parameter and $ \beta $ is a learnable shift parameter.
        
        LLaMA favours RMS normalisation as it requires less computation, we are not computing $ \mu $ or $ \sigma $, and it works well in practice.
    
    - Self-Attention Layers: 
    
    - We have text in its normalised form as an input sequence. Self-attention is the pairwise interdependence of all elements composing an input. $$\mathbf{X} = [x_1, x_2, ..., x_n] $$, where each $ x_i $ is a vector representing a normalised word or token in the sequence. Here's a detailed explanation of how self-attention works with regards to LLaMA LLMs, including the mathematical formulas involved:

        - 1. Calculating Query, Key, and Value Vectors:
   
            The self-attention mechanism then three sets of vectors for each input vector: 
            - Query (Q): Represents the importance of the current element (given from the encoded part of the source sentence)
            - Key (K): Represents the relationships of the current element with other elements (given from the encoded part of the source sentence)
            - Value (V): Holds the information to be passed to the next layer (given from the encoded part of the target sentence)
            
            They are computed as follows:
            $$ Q = \text{Normalized Input} \cdot W^Q $$
            $$ K = \text{Normalized Input} \cdot W^K $$
            $$ V = \text{Normalized Input} \cdot W^V $$
            Where $ W^Q $, $ W^K $, and $ W^V $ are weight matrices that are learned during training, and  $\text{Normalized Input}$ is the vector of normalised inputs caluclated above during the RMS normalisation step.
            
        - 2. Applying Rotary Embeddings to Q and K. 
            The theory and mathematical formulas regarding this figure on Su et al. 2021 paper "ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING", in the 3.2.2 sub-chapter intiled "General Form" on page 5 of the paper. To put it simply, the query and key vectors are multiplied by a sparse rotary matrix, that depends on the vectors dimentions, with pre-defined parameters. 
        
        - 3. Attention Scores:
            
            The attention scores, are then computed as such:
        
            $$ \text{Attention Score} = \text{Softmax}(\frac{QK^T}{\sqrt{d_k}})V $$
            such that $ d_k $ is the dimensionality of the Key vectors, the Q and K values have been applied rotary embeddings, and the softmax function is defined as: $$ \text{Softmax}(\frac{QK^T}{\sqrt{d_k}}) = \frac{\exp(\frac{QK^T}{\sqrt{d_k}})}{\sum \exp(\frac{QK^T}{\sqrt{d_k}})} $$
            
            This step captures how much attention each element should pay to others. The raw attention scores are then scaled to mitigate issues related to the dimension of the vectors.
            
         - 3.5 KV cache:
         
             Given we are dealing with a LLaMA model, a process called KV cache is also used in the attention score calculations. The refers to a mechanism that optimizes the computation of attention scores by reusing previously computed key (K) and value (V) pairs. 
             
             The key idea behind the "KV cache" is to store the key-value pairs for each position in the sequence during the computation of attention scores. This stored information can be reused when computing the attention scores for subsequent positions in an autoregressive decoding scenario. Autoregressive decoding involves generating one token at a time in a sequential manner, and during this process, attention is repeatedly calculated for each new token. By caching the key-value pairs and reusing them for positions that have already been processed, the transformer avoids redundant computations and significantly speeds up the attention calculation. The "KV cache" therefore helps to make the attention score computation more efficient and reduces the overall computational cost.
            
        - 4.Grouped Multi-Query Attention (with KV cache):
        
            LLaMA models use a process called Grouped Multi-Query Attention to reduce computing time. Grouped Multi-Query Attention is an extension of the traditional attention mechanism in transformers, specifically designed to further enhance computational efficiency, particularly in autoregressive decoding scenarios. This mechanism builds upon the concept of Multi-Query Attention with KV (Key-Value) cache, which optimizes attention calculations by reusing previously computed key-value pairs.

            In Grouped Multi-Query Attention, the queries are grouped and processed together, further reducing the computational cost. Here's how it works:

            - Initialization: During the initial computation of attention for the first position in the sequence, the key (K) and value (V) pairs are computed and stored in a cache.

            - Grouping Queries: Instead of processing each query independently, queries are grouped together. This means that instead of attending separately to all keys for each individual query, a group of queries attends to the same set of keys.

            - Efficient Computation: The attention mechanism then computes the attention scores for all the grouped queries simultaneously, utilizing the cached key-value pairs. This grouped computation is more efficient than processing each query independently.
            
            - Update Cache: After the grouped attention is computed, the updated key-value pairs are stored back in the cache. These updated key-value pairs can be reused for subsequent groups of queries.

            The grouping of queries allows for parallelization of attention calculations, making the process more efficient, especially in scenarios where there is a high degree of sequential dependence, such as autoregressive decoding. Here's a visualisation illustrating Grouped Multi-Query Attention side-by-side Multi-Head Attention (used in traditional transformer architecture) and Multi-Query: 
            ![](https://production-media.paperswithcode.com/methods/9de342b2-d0f9-4a96-8a18-5700ba11a42b.png)
           
            Grouped Multi-Query Attention is an optimization that leverages the benefits of both grouping queries and reusing key-value pairs from the cache. It is particularly useful in autoregressive decoding scenarios, where the efficiency of attention calculations is crucial for generating sequences in a timely manner. It is important to note that multi-query attention does degrade the overall quality of the model, but only slightly; this degradation is worth it for the impoved computational costs. This grouping technique is a good in-between multi-head and multi-query attention. 

            In summary, the self-attention layer in transformers enables the model to weigh the importance of each element in a sequence concerning all other elements. This mechanism is a key innovation that contributes to the success of transformers in capturing contextual information and dependencies in various natural language processing and sequential data tasks.
            
    - RMS Layer Normalization of the Multi-Head Self-Attention Layer's Output:
    
        Before going through a feedforward neural network, the output of the self-attention layers is once again pre-normalised and goes through an RMS function. The mathematical formulas involved are the same as previously laid-out.

    - Feedforward Neural-Networks:
    
        - The feedforward layer in a transformer encoder is a key component responsible for transforming the representations obtained from the self-attention mechanism into a more abstract and compact form. The feedforward layer is applied after the concatenation of the outputs from the multiple attention heads. In the case of LLaMA LLMs, here's how it works:
                
        - 1. First Fully Connected Layer:
            The normalized input is passed through the first fully connected layer with a SwiGLU activation and produces an intermediate output as such:
            $$ \text{Intermediate Output} = \text{SwiGLU}(\text{Linear}(\text{Normalized Input})) $$
            Here, "Linear" represents a fully connected layer with learnable weights and biases, "SwiGLU" is the Swish-Gated Linear Unit activation function, which is defined as $ \text{SwiGLU}(x) = x \cdot \sigma (x) $ where $ \sigma $ is the sigmoid activation function.  

        - 2. Second Fully and Third Connected Layers:
            The intermediate output is then passed through second and third fully connected layers without any activation. We have: $$ \text{Final Output} = \text{Linear}(\text{Intermediate Output}) $$, such that the "Linear" function is defined as $\text{Linear}(X) = XW + b $, where X is the input tensor, W the weight matrix and b the bias vector.

        - 3. Residual Connection and Layer Normalization:

            The final output is added to the original input (residual connection) to facilitate the flow of information through the network. $$ \text{Feedforward Output} = \text{LayerNorm}(\text{Input} + \text{Final Output}) $$ such that the "LayerNorm" function is the same as defined above.
        
            The feedforward layer is crucial for capturing complex patterns and relationships within the input sequence, contributing to the expressive power of the transformer model. The use of residual connections and layer normalization helps stabilize training and allows for more effective learning. 

        Similarly to the traditional transformer architecture, the enoder is repeated multiple times to create a deep hierarchical structure. Indeed, repeating the encoder multiple times in a Transformer architecture allows the model to learn hierarchical representations, capture complex dependencies, and generalize well to a variety of tasks. The depth of the encoder is a crucial hyperparameter that is often tuned based on the complexity of the learning task and the available computational resources.
        
        
- **Final RMS Layer Normalization**: The output of the encoder part of the transformer architecture is a set of high-dimensional, contextualized embeddings representing each token in the input sequence. This output is normalized using RMS Layer Normalization; the mathematical formulas involved are the same as previously laid-out for RMS Normalization.

- **Final Linear Layer**: The output of the Final RMS Layer is then put through a final linear layer; the mathematical formulas involved are the same as previously laid-out for linear layers.

- **Softmax Activation Function**: The output of Final Linear Layer is finally passed through a softmax activation function to ensure that the final outputs sum to 1. Indeed, the softmax function normalizes the logits, converting them into probabilities. Each probability in the output corresponds to the likelihood of the corresponding word/token in the vocabulary. The final output of the model is therefore a sequence of probabilities for each position in the output sequence. During training, the model is typically trained to minimize the negative log-likelihood of the target sequence. During inference or generation, the model can sample or select the most likely tokens based on the probabilities.

Finally, given all this information, we can draw is a schema of the LLaMA LLM transformer architecture:

![LLaMA%20Transformer.png](attachment:LLaMA%20Transformer.png)     

# 8-bit quantization


Large language models (LLMs) like LLaMA and LLaMA 2 pose a challenge due to substantial GPU memory requirements, especially with models growing to billions of parameters. To address this, 8-bit quantization, in particular with the LLM.int8() function, offers a solution.

Before delving into 8-bit quantization, it's essential to understand data precision in machine learning. This process involves transforming high-precision floating-point data (32 bits) into a more compact 8-bit format (int8), optimizing model size and computational efficiency without compromising inference quality.

In the landscape of floating-point data types, Float32 (FP32) proves cumbersome, particularly in billion-parameter models. Float16 (FP16) and bfloat16 (BF16), 16-bit representations, offer size reduction but come with precision trade-offs.

8-bit quantization compresses model weights by achieving near-identical inference outcomes with 16-bit and 8-bit precision, effectively halving and quartering the model size. This involves rounding from higher precision (FP32) to lower precision (int8), introducing quantization error - a 'lossy compression' impacting model performance.

Two common 8-bit quantization techniques, zero-point and absolute maximum (absmax), scale input data by a quantization constant, mapping original values into the reduced int8 range. This results in a significantly smaller model size, though with potential information loss due to the noisiness of the quantization process. However, the LLM.int8() implementation introduced by Tim Dettmers et al. (2022) in "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", is the first technique that does not degrade performance even for large models with up to 176 bilion parameters. 

## LLM.int8()

The LLM.int8() algorithm itself seeks to complete the matrix multiplication computation in three steps:

- From the input hidden states, the outliers are extracted (i.e. values that are larger than a certain threshold) by column. Here, the hidden states refer to the intermediate representations or features generated by the model's feedforward layers during its computation.

- The matrix multiplication is done on the outliers in FP16 and on the non-outliers in int8. The matrix multiplication refers to the multiplication of the hidden states by a weight matrix.

- The non-outlier results are dequantized and both outlier and non-outlier results are added together to receive the full result in FP16. In other words, dequantizing the non-outlier results involves converting the int8 values back to a higher precision format, here FP16. The dequantized non-outlier results are then added to the FP16 results obtained from the outliers. This step synthesizes the information from both sets of computations to generate the final result in FP16.

These steps can be summarised by the following visualisation:

![](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/Matmul.png)

In essence, once the hidden states are computed, the outliers are extracted using a custom threshold, and the matrix is decomposed into two parts as explained above. Tim Dettmers et al. found that extracting all outliers with magnitude 6 or greater in this way recoveres full inference performance. The outlier extraction is done in fp16 so it is a classic matrix multiplication, whereas the 8-bit matrix multiplication is done by quantizing the weights and hidden states into 8-bit precision using vector-wise quantization - that is, row-wise quantization for the hidden state and column-wise quantization for the weight matrix; the mathematical formulas involved for this step are specified on the schea above. After this step, the results are dequantized and returned in half-precision in order to add them to the first matrix multiplication.




```python

```
