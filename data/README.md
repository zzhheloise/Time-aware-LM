# WMT News Crawl

We downloaded document-split versions of the English WMT News Crawl dataset. Its time period range from 2007 to 2021. As the dataset does not provide document IDs, we used [SHA256 hashes](https://github.com/deepmind/deepmind-research/tree/master/pitfalls_static_language_models) of the Base64 encoded unsplit texts of articles as their IDs.

We use WMT news to train T5 models with parallel LoRAs. We inject LoRA-2016, LoRA-2017, LoRA-2018, LoRA-2019, LoRA-2020 and LoRA-future into T5 model. For example, we train LoRA-2016 on 2016 WMT news, during which we freeze parameters of the encoder for T5, following the setup of Jang et al. (2021). 

See the details of downloading and preprocessing WMT news in `\data\WMT-news`

# LAMA

## TempLAMA

## LAMA in CKL

LAMA is used for task1
