# LLM-Quantization

This project aims to optimize the inference process of Large Language Models (LLM), particularly the LLAMA 2 model developped by Meta, by implementing and analyzing 8-bit quantization techniques. The project will explore the how transformers and causal transformers function with regards to LLaMA models, as well as the hardware and computational aspects of the the LLAMA 2 model, focusing on reducing computational time while maintaining model performance.

This project is comprised of:
    
- REPORT.md: A technical report detailing the theoretical aspects of transformers within LLaMA large language models, and 8-bit quantization. A detailed explanation of how transformers and causal transformers function within LLaMA models. The concept of 8-bit quantization is also explained. 

- 5 codebases:

    - huggingface_int.py: 
        The code needed to run a side-by-side comparison of the time spent for a LLaMA 1 model trained on 7B parameters to run and answer a promt, without 8-bit quantization and with 8-bit quantization.
        
    - LLaMa_model.py:
        A regular LLaMA 2 model architecture constructed from scratch, with help of some ressouces listed below.
        
    - inference.py: 
        The inference code to load the weights into the LLaMA 2 model "LLaMa_model". Some customized prompts are set at the end to evaluate the computing time.
        
    - LLaMa_model_8bit.py: 
        A LLaMA 2 model architecture with 8-bit quantization. The hidden linear layers of the regular LLaMA 2 model were replaced with 8-bit optimizer layers. 
   
    - inference8bit.py:
        The inference code to load the weights into the LLaMA 2 model "LLaMa_model_8bit". Some customized prompts are set at the end to evaluate the computing time.
    
    - tokenizer.py: 
        Tokenizer model, as found on the "FacebookReaserch" GitHub.
        
Requirements:

- Python Packages:
    - torch
    - sentencepiece
    - tqdm
    - sentencepiece

- LLaMA 2 - 7B model for inference. This model can be found at "https://ai.meta.com/llama/". The model in question should be requested, after which Meta should send an email with a confirmation link. The model is then downloadable via the computer's terminal using said confirmation link. The resulting file should be called 'consolidated.00.pth'. 



