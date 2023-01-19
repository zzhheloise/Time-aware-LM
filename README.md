# Time-aware-LM

The goal of this project is to test if injecting LoRAs into T5 models could mitigate 'catastrophic forgetting' problem and/or make LMs perform better on predicting future utterances.

We aim to answer two questions: 

(1) Can parallel LoRAs mitigate "catastrophic forgetting" problem? This is, can T5 model with parallel LoRAs forget past knowledge more slowly during being fine-tuned on new corpora, compared with that without parallel LoRAs? 

(2) How to design LoRA-future to tame the model to predict future utterance with higher accuracy (lower perplexity)? Moreover, can LoRA-future help T5 model perform better on future closed-book questions? 

## Experiment

### LoRA setup

### Control setup

## Evaluation

### Task 1: forget past knowledge


### Task 2: relative perplexity

task 3: future closed-book qa

## References

Bhuwan Dhingra, Jeremy R. Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, and William W. Cohen (2022). Time-Aware Language Models as Temporal Knowledge Bases. ACL.

Jang, Joel and Ye, Seonghyeon and Yang, Sohee and Shin, Joongbo and Han, Janghoon and Kim, Gyeonghun and Choi, Stanley Jungkyu and Seo, Minjoon (2022). Towards Continual Knowledge Learning of Language Models. ICLR.

Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig (2022). Towards a Unified View of Parameter-Efficient Transfer Learning. ICLR.
