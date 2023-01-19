# Time-aware-LM

We aim to answer two questions: 

(1) Can parallel LoRAs mitigate "catastrophic forgetting" problem? This is, while being fine-tuned on new corpora, can T5 models with parallel LoRAs forget past knowledge more slowly compared with those without parallel LoRAs? 

(2) Can T5 models with parallel LoRAs and LoRA-future perform better on predicting future utterance? How to design LoRA-future to tame the model to predict future utterance? Moreover, can LoRA-future help T5 model perform better on future closed-book questions? 

## Dataset

See `\data` for details

## Experiment

We perform our experiments with an encoder-decoder model, T5, a large LM (about 737M params) initially pretrained on April 2019 dump of C4 and May 2020 dump of Wikipedia with salient span masking (SSM).

### LoRA setup

Low-Rank Adaptation, or LoRA (Hu et al. 2021) freezes the pre-trained model weights and injects trainable rank decomposition matrics into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Following Hu et al. (2021), we apply LoRAs to T5.

Jang et al. (2021) injects two LoRAs to T5 for creating a ever-changing LM and finds that LMs are prone to more forgetting as they go through multiple traning phrases. We injects 6 LoRAs to T5, namely 

### Control setup

## Evaluation

### Task 1: forgeting

### Task 2: predicting

relative perplexity

task 3: future closed-book qa

## References

Bhuwan Dhingra, Jeremy R. Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, and William W. Cohen (2022). Time-Aware Language Models as Temporal Knowledge Bases. ACL.

Jang, Joel and Ye, Seonghyeon and Yang, Sohee and Shin, Joongbo and Han, Janghoon and Kim, Gyeonghun and Choi, Stanley Jungkyu and Seo, Minjoon (2022). Towards Continual Knowledge Learning of Language Models. ICLR.

Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig (2022). Towards a Unified View of Parameter-Efficient Transfer Learning. ICLR.

Angeliki Lazaridou and Adhiguna Kuncoro and Elena Gribovskaya and Devang Agrawal and Adam Liska and Tayfun Terzi and Mai Gimenez and Cyprien de Masson d'Autume and Tomas Kocisky and Sebastian Ruder and Dani Yogatama and Kris Cao and Susannah Young and Phil Blunsom (2021). Mind the Gap: Assessing Temporal Generalization in Neural Language Models. NeurIPS.

Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Weizhu Chen (2021). LoRA: Low-Rank Adaptation of Large Language Models. ArXiv.
