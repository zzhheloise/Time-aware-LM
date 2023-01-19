# WMT News Crawl

We downloaded document-split versions of the English WMT News Crawl dataset. Its time period range from 2007 to 2021. As the dataset does not provide document IDs, we used [SHA256 hashes](https://github.com/deepmind/deepmind-research/tree/master/pitfalls_static_language_models) of the Base64 encoded unsplit texts of articles as their IDs.

We use WMT news to train a T5 model with parallel LoRAs. We inject LoRA-2016, LoRA-2017, LoRA-2018, LoRA-2019, LoRA-2020 and LoRA-future into the encoder of this T5 model. For example, we train LoRA-2016 on 2016 WMT news, during which we freeze parameters of the encoder for T5, following the setup of Jang et al. (2021). Aftering training LoRA-2016, we then train the LoRA of next year, until finishing training LoRA-2020. We would save 5 finetuned T5 models after 5 training phrases.

For Control Group, we repeat above training method on a T5 model without LoRAs. The difference is that we finetune all parameters in Control Group.

As for how to train LoRA-future, we discuss this in LAMA part.

Moreover, we use 2021 WMT news for evaluation. We sub-sample a test set of 12k test documents (1k per test month). We use 10 finetuned T5 models (LoRA group and Control group) to calculate their perplexity results of predicting 2021 utterance. We expect that models deteriorates more as we ask it to predict data further away from the training period, and that models with LoRAs predict future data more accurately than their counterparts without LoRAs.

See the details of downloading and preprocessing WMT news in `\data\WMT-news`

# LAMA

LAnguage Model Analysis (LAMA) task requires probing LMs for world knowledge in a zero-shot manner through slot-filling. We use TempLAMA and InvariantLAMA to fintune LoRA-future and evaluate the forgetting problem of fintuned T5 models.

## TempLAMA

Dhingra et al. (2022) identify all facts which have either a start or an end date after 2010 and whose subjects and objects are both entities with Wikipedia pages. For each subject and each relation they gather all the objects with their associated time interval and construct a separate query for each year in that interval. In total they construct 50,310 queries across 11 years.

We use TempLAMA to train LoRA-future. After training LoRA-2016, we want to design a dataset to train LoRA-future, in order to inform this model of future knowledge. We can use the queries in 2017 in TempLAMA to train LoRA-future, using template like "In next year, Subject works for _X_". 

We expect that models with trained LoRA-future can perform the best on predicting future utterance among all trained models. This is because we believe "Time is continuous". Pretrained language models know the knowledge in the next year can somehow understand the trend of chaning facts, meaning it may predict knowledge in next 3 years better than the model not knowing future knowledge at all.

See the details of downloading and preprocessing TempLAMA in `\data\TempLAMA`.

## InvariantLAMA

Jang et al. (2021) created InvariantLAMA, a subset of the LAMA task for measuring time-invariant knowledge which might be forgetten during CKL. We use InvariantLAMA to evaluate the forgetting problem of finetuned T5 models. We expect that as models forget more past facts as they go through more training phrases, and that models with LoRAs forget past facts more slowly than their counterparts without LoRAs.

However, the problem is that we need to check if the knowledge of InvariantLAMA is before 2016. If InvariantLAMA is updated to new knowledge after 2016, T5 model that is only trained on 2016 WMT news performs poorly on InvariantLAMA not beacause of forgetting, but because of not being able to predict the future knowledge. We may need to work on a new InvariantLAMA if this problem do exist.

See the details of downloading InvariantLAMA in `\data\InvariantLAMA`.
