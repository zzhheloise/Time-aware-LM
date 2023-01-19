import gzip
import hashlib
import base64
from re import T
import pandas as pd
from scipy import stats
import numpy as np
import time
import os
import torch
import json
import pprint
from transformers import T5Tokenizer
import spacy

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")
tokenizer = T5Tokenizer.from_pretrained('/root/zhihan/t5-large-ssm') # google/t5-large-ssm
sentinels=[]
for i in range(500): #zzh 原代码是200，t5-large-ssm的tokenizer_config.json是100，原因是什么？这个数字可以随意修改吗？
    sentinels.append(f'<extra_id_{i}>')

def ssm(index, text):
    if text=='':
        return None
    input = ""
    target = ""
    sentinel_cnt=0
    previous_end=0
    doc = nlp(text)
    for ent in doc.ents:
        start_index = ent.start_char
        end_index = ent.end_char
        word = ent.text
        #print(sentinel_cnt)
        input = input + text[previous_end:start_index] + sentinels[sentinel_cnt]
        target = target + sentinels[sentinel_cnt]+" " + word +" "
        previous_end = end_index
        sentinel_cnt+=1
    input = input + text[previous_end:]
    target = target + sentinels[sentinel_cnt]
    #print(index)
    return index, text, input, target

def wmt_preprocess(file):
  with gzip.open(file) as gz_file: #gzip.open #'news-docs.2009.en.filtered.gz'
    for line in gz_file:
      date, sentence_split_text, unsplit_text = line.decode('utf-8').strip().split('\t') #.encode('utf-8')
      docid = hashlib.sha256(unsplit_text.encode('utf-8')).hexdigest()
      sentence_split_text = base64.b64decode(sentence_split_text)
      unsplit_text = base64.b64decode(unsplit_text)
      yield docid, (date, sentence_split_text, unsplit_text)


length_limit = 250 #The limit of words per input
file_list = ['2016','2017','2018','2019','2020','2021']
#zzh full list: '2007','2008','2009', '2010', '2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021'

for year in file_list:
    file_name = 'news-docs.'+year+'.en.filtered.gz'
    recent_news = []
    article_index = 0
    for docid, (date, sentence_split_text, unsplit_text) in wmt_preprocess(file_name):
        article_index+=1
        prtstr = str(year)+' '+str(article_index)
        print(prtstr) #zzh
        #if article_index==20000: #zzh 原本为20000，目前可以限制，后面再跑实验可以调大
        #    break
        text = str(unsplit_text, encoding='utf-8')
        #tokens = tokenizer.encode(text) #zzh
        #token_len = len(tokens) #zzh
        text = text.replace('\n', '')
        if len(text.split()) > length_limit:
            word_list = text.split()
            seg1 = word_list[:length_limit]
            try:
                segment1, seg2_a = (' '.join(seg1)).rsplit('.',1)
            except ValueError as e:
                seg2_a = ''
            segment2 = seg2_a + (' '.join(word_list[length_limit:]))
            output = ssm(article_index, segment1)
            if output: recent_news.append(output)

            while(len(segment2.split()) > length_limit):
                word_list = segment2.split()
                seg1_ = word_list[:length_limit]
                if '.' in ' '.join(seg1_):
                    segment1_, seg2_a_ = (' '.join(seg1_)).rsplit('.',1)
                    segment2 = seg2_a_ + (' '.join(word_list[length_limit:]))
                else:
                    segment1_ = ' '.join(seg1_)
                    segment2 = (' '.join(word_list[length_limit:]))
                output = ssm(article_index, segment1_)
                if output:  recent_news.append(output)
        else:
            output=ssm(article_index, text)
            if output: 
                recent_news.append(output)
    save_name = '/root/zhihan/ckl-dataset/WMTNews-filtered/news-filtered-' + year + '-0113.csv'
    pd.DataFrame(recent_news, columns=['index','original', 'input', 'output']).to_csv(save_name)