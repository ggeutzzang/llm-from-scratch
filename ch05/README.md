# 5장: 레이블이 없는 데이터를 활용한 사전 훈련

&nbsp;
## 예제 코드

- [01_main-chapter-code](01_main-chapter-code)에는 5장의 코드와 연습 문제 해답이 포함되어 있습니다. 

&nbsp;
## 보너스 자료

- [02_alternative_weight_loading](02_alternative_weight_loading) contains code to load the GPT model weights from alternative places in case the model weights become unavailable from OpenAI
- [03_bonus_pretraining_on_gutenberg](03_bonus_pretraining_on_gutenberg) contains code to pretrain the LLM longer on the whole corpus of books from Project Gutenberg
- [04_learning_rate_schedulers](04_learning_rate_schedulers) contains code implementing a more sophisticated training function including learning rate schedulers and gradient clipping
- [05_bonus_hparam_tuning](05_bonus_hparam_tuning) contains an optional hyperparameter tuning script
- [06_user_interface](06_user_interface) implements an interactive user interface to interact with the pretrained LLM
- [07_gpt_to_llama](07_gpt_to_llama) contains a step-by-step guide for converting a GPT architecture implementation to Llama 3.2 and loads pretrained weights from Meta AI
- [08_memory_efficient_weight_loading](08_memory_efficient_weight_loading) contains a bonus notebook showing how to load model weights via PyTorch's `load_state_dict` method more efficiently
- [09_extending-tokenizers](09_extending-tokenizers) contains a from-scratch implementation of the GPT-2 BPE tokenizer
- [10_llm-training-speed](10_llm-training-speed) shows PyTorch performance tips to improve the LLM training speed
- [11_qwen3](11_qwen3) A from-scratch implementation of Qwen3 0.6B and Qwen3 30B-A3B (Mixture-of-Experts) including code to load the pretrained weights of the base, reasoning, and coding model variants
- [12_gemma3](12_gemma3) A from-scratch implementation of Gemma 3 270M and alternative with KV cache, including code to load the pretrained weights
