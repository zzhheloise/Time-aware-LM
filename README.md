# Time-aware-LM

We aim to answer two questions: 

(1) Can parallel LoRAs mitigate "catastrophic forgetting" problem? This is, while being fine-tuned on new corpora, can T5 models with parallel LoRAs forget past knowledge more slowly compared with those without parallel LoRAs? 

(2) Can T5 models with parallel LoRAs and LoRA-future perform better on predicting future utterance? How to design LoRA-future to tame the model to predict future utterance? Moreover, can LoRA-future help T5 model perform better on future closed-book questions? 

## Dataset

See `\data` for details

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

Angeliki Lazaridou and Adhiguna Kuncoro and Elena Gribovskaya and Devang Agrawal and Adam Liska and Tayfun Terzi and Mai Gimenez and Cyprien de Masson d'Autume and Tomas Kocisky and Sebastian Ruder and Dani Yogatama and Kris Cao and Susannah Young and Phil Blunsom (2021). Mind the Gap: Assessing Temporal Generalization in Neural Language Models. NeurIPS.

Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Weizhu Chen (2021). LoRA: Low-Rank Adaptation of Large Language Models. ArXiv.
