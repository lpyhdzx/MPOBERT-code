# Enhancing Scalability of Pre-trained Language Models via Efficient Parameter Sharing

This is the implementation of the paper:
> Peiyu Liu, Ze-Feng Gao, Yushuo Chen, Wayne Xin Zhao and Ji-Rong Wen. Enhancing Scalability of Pre-trained Language Models via Efficient Parameter Sharing
*Updates*:

* [October 14] We update the README. The code is coming soon!

---
## Abstract
In this paper, we propose a highly parameter-efficient approach to scaling pre-trained language models~(PLMs) to a deeper model depth. 
Unlike prior work that shares all parameters or uses extra blocks, we design a more capable parameter-sharing architecture based on  matrix product operator~(MPO), an efficient tensor decomposition method to factorize the parameter matrix into a set of local tensors. Based on such a decomposition,  we share the important local tensor across all layers for reducing the model size and meanwhile keep layer-specific tensors~(also using Adapters) for enhancing the adaptation flexibility. To improve the model training, we further propose a stable initialization algorithm tailored for the MPO-based architecture. Extensive experiments have demonstrated the effectiveness of our proposed model in enhancing scalability and achieving higher performance (i.e., with fewer parameters than BERT_base, we successfully scale the model depth by a factor of 4x and even achieve 0.1 points higher than BERT_large for GLUE score). All the experimental codes will be released after the review period.s lights on the possibilities of extremely low-bit quantization for LLMs.

## Pre-training

## Fine-tuning

## TODO
- [ ] Add the code for pretraining
- [ ] Add the code for fine-tuning
