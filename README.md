
# vLLM Iteration-Level Schedule Optimization
#### Ari Singer, Jack Holland and Vishwa Ramanakumar
#### University of Michigan — EECS 598: Systems for Generative AI

This repository holds our custom vLLM implementation, as well as the Jupyter notebook used for obtaining our results. To run, you will need to use Google Colab with an A100, as other instances will not fit the LLM we use. Please see instructions in the notebook for use.

### Abstract
In recent years, considerable resources have been put into developing powerful transformer-based generative large language models (e.g. GPT-3). Given the auto-regressive nature of these models, when processing an inference request, multiple model calls need to be made to generate one subsequent token at a time and complete one iteration. Recent inference serving algorithms such as vLLM [Kwon et al., 2023](https://arxiv.org/abs/2309.06180), have made significant progress in optimizing inference latency and throughput with the use of iteration-level batching. However, iteration-level batching can fall victim to certain bottleneck inefficiencies when variance in request context length is high because longer requests have longer token generation time. To mitigate this, we propose an updated scheduler algorithm built on vLLM which preempts requests that exceed a certain preemption threshold, and runs them separately. Our work exhibits improvements in average token generation time and also shows potential for future work. 




