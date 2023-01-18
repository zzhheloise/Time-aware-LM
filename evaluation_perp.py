
import gzip
import hashlib
import base64
import torch
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
from itertools import chain
from Datasets import Pretrain
from torch.utils.data import DataLoader
import csv
import os

def wmt_preprocess(file):
    with gzip.open(file) as gz_file:
        for line in gz_file:
            date, sentence_split_text, unsplit_text = line.decode('utf-8').strip().split('\t')
            docid = hashlib.sha256(unsplit_text.encode('utf-8')).hexdigest()
            sentence_split_text = base64.b64decode(sentence_split_text)
            unsplit_text = base64.b64decode(unsplit_text)
            yield docid, (date, sentence_split_text, unsplit_text)

def evaluate_perp(args, Model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    '''
    model = Model(args)
    if args.checkpoint_path!="":
        model = Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args, strict=False)
    model.eval()
    model.to('cuda')
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    '''

    month_tag = []
    full_index = []
    local_index = []
    wmt_text = []

    month = 2
    flag = 0
    for docid, (date, sentence_split_text, unsplit_text) in wmt_preprocess(args.dataset):
        flag += 1
        if int(date[4:6]) == month:
            month_tag.append(flag)
            month += 1
    month_tag.append(flag)

    random.seed(10)
    for i in range(len(month_tag)):
        if i == 0:
            local_index = range(0,month_tag[i])
        else:
            local_index = range(month_tag[i-1],month_tag[i])
        local_index = random.sample(local_index, 1000)
        full_index.append(local_index)
    full_index = list(chain.from_iterable(full_index))

    #zzh clean_up() is taken from evaluation_lama.py, think if this function is needed in this file
    def clean_up(text): 
        text =text.replace('<pad>', '')
        text = text.replace('</s>', '')
        text =text.replace('\n', '')
        text = text.replace(".", '')
        text = text.replace(',', '')
        text = text.replace("'", '')
        text = text.replace('"', '')
        return text 

    flag =0
    for docid, (date, sentence_split_text, unsplit_text) in wmt_preprocess(args.dataset):
        flag += 1
        if (flag in full_index):
            text = str(unsplit_text, encoding='utf-8')
            #wmt_text.append(text.replace('\n',' '))
            wmt_text.append(clean_up(text)) #zzh
        else:
            continue

    encodings = tokenizer("\n\n".join(wmt_text), return_tensors="pt")
    max_length = model.config.d_model #zzh 原本为model.config.d_model (/n_positions), model指t5-large-ssm，我们直接从config文件中找到该值1024
    stride = 512
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to('cuda')
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

  
    #zzh taken from evaluation_lama.py: If folder doesn't exist, then create it.
    MYDIR = ("/".join((args.output_log.split('/'))[:-1]))
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    else:
        print(MYDIR, "folder already exists.")
    
    with open(args.output_log, 'a', newline='') as writefile:  
        writer = csv.writer(writefile)
        writer.writerow([args.dataset, args.checkpoint_path, ppl])
        print(f'ppl: {ppl}')