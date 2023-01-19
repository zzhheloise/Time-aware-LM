# Time-aware-LM

We aim to answer two questions: 

(1) Can parallel LoRAs mitigate "catastrophic forgetting" problem? This is, while being fine-tuned on new corpora, can T5 models with parallel LoRAs forget past knowledge more slowly compared with those without parallel LoRAs? 

(2) Can T5 models with parallel LoRAs and LoRA-future perform better on predicting future utterance? How to design LoRA-future to tame the model to predict future utterance? Moreover, can LoRA-future help T5 model perform better on future closed-book questions? 

## Dataset

Include WMT News Crawl Dataset, TempLAMA and InvariateLAMA .See `\data\README.md` for details

## Experiment

We perform our experiments with an encoder-decoder model, T5, a large LM (about 737M params) initially pretrained on April 2019 dump of C4 and May 2020 dump of Wikipedia with salient span masking (SSM).

### LoRA setup

Low-Rank Adaptation, or LoRA (Hu et al. 2021) freezes the pre-trained model weights and injects trainable rank decomposition matrics into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Jang et al. (2021) injects two LoRAs to T5 for creating a ever-changing LM and finds that LMs are prone to more forgetting as they go through multiple traning phrases. Following above two works, we injects 6 LoRAs to T5, namely LoRA-2016, LoRA-2017, LoRA-2018, LoRA-2019, LoRA-2020 and LoRA-future.

We use WMT News Crawl Dataset from 2016 to 2020, to continuously train the T5 model with parallel LoRAs. For example, we train LoRA-2016 on 2016 WMT news, during which we freeze parameters of the encoder for T5. We then train the LoRA of next year, until finishing training LoRA-2020. We would save 5 finetuned T5 models after 5 training phrases.

```
python run.py --config t5-train.json
```

We use TempLAMA to train LoRA-future. After training LoRA-2016, we want to design a dataset to train LoRA-future, in order to inform this model of future knowledge. We can use the queries in 2017 in TempLAMA to train LoRA-future, using template like "In next year, Subject works for X".

### Control setup

The only difference between two setups is that Control group train T5 models without any LoRAs. The training process is exactly the same.

## Evaluation

### Task 1: forgeting

We use InvariantLAMA to evaluate the forgetting problem of finetuned T5 models.

We expect that as models forget more past facts as they go through more training phrases, and that models with LoRAs forget past facts more slowly than their counterparts without LoRAs.

### Task 2: predicting

We use 2021 WMT news for evaluating the predicting ability of models. We sub-sample a test set of 12k test documents (1k per test month). We use 10 finetuned T5 models (LoRA group and Control group) to calculate their perplexity results of predicting 2021 utterance. 

Following Lazaridou et al. (2021), we use Relative Perplexity changes (%) between the model in Control group and that in LoRA groups, whose training phrases are the same. This is because some months have longer documents, which leads to higher absolute perplexity. The calculation formula of Relative Perplexity changes (%) is that the difference of Absolute Perpelexity of Control model minus that of LoRA model divided by that of LoRA model

We expect that models deteriorates more as we ask it to predict data further away from the training period, and that models with LoRAs predict future data more accurately than their counterparts without LoRAs. This is, Relative Perplexity change is always positive, and that it is shows upward slope whose x-axis is the end date of each traning phrases.

Moreover, we also consider evaluating through answering future closed-book questions. We will add more details later.

## References

Bhuwan Dhingra, Jeremy R. Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, and William W. Cohen (2022). Time-Aware Language Models as Temporal Knowledge Bases. ACL.

Jang, Joel and Ye, Seonghyeon and Yang, Sohee and Shin, Joongbo and Han, Janghoon and Kim, Gyeonghun and Choi, Stanley Jungkyu and Seo, Minjoon (2022). Towards Continual Knowledge Learning of Language Models. ICLR.

Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig (2022). Towards a Unified View of Parameter-Efficient Transfer Learning. ICLR.

Angeliki Lazaridou and Adhiguna Kuncoro and Elena Gribovskaya and Devang Agrawal and Adam Liska and Tayfun Terzi and Mai Gimenez and Cyprien de Masson d'Autume and Tomas Kocisky and Sebastian Ruder and Dani Yogatama and Kris Cao and Susannah Young and Phil Blunsom (2021). Mind the Gap: Assessing Temporal Generalization in Neural Language Models. NeurIPS.

Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Weizhu Chen (2021). LoRA: Low-Rank Adaptation of Large Language Models. ArXiv.
