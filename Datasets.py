from torch.utils.data import Dataset
import pandas as pd
import json
import random

from datasets import load_dataset

class Pretrain(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, args, length=None):
        self.args = args
        self.tokenizer = tokenizer
        self.type_path = type_path
        self.ssm = False
        if 't5' in args.model_name_or_path:
            self.model_type='T5'
        elif 'gpt2' in args.model_name_or_path:
            self.model_type='GPT2'
        ids_to_answers = None
        # dataset for continual training
        if 'wmt' or 'WMT' in self.args.dataset:
            self.dataset = pd.read_csv(self.args.dataset)
        # dataset for evaluation
        else: 
            if self.args.dataset == 'invariantlama':
                self.dataset = pd.read_csv('data/InvariantLAMA/invariantLAMA.csv')
            elif self.args.dataset == 'templama':
                rp_dir = 'data/TempLAMA/'
                file_list = ["train.json", "test.json", "val.json"]
                templama_data = []
                for file_name in file_list:
                    file = open(rp_dir + file_name, 'r', encoding='utf-8')
                    for line in file.readlines():
                        dic = json.loads(line)
                        templama_data.append(dic)
                if int(self.args.method[-2:]) < 10 :
                    target_year = '2010'
                else:
                    target_year = '20' + self.args.method[-2:]
                yearlama = pd.DataFrame(columns=['id','input','output'])
                id = 0
                for i in range(len(templama_data)):
                    input = templama_data[i]['query'].replace('_X_','<extra_id_0>')
                    output = templama_data[i]['most_recent_answer']['name']
                    year = templama_data[i]['date']
                    if year == target_year:
                        yearlama.loc[id] = [id, input, output]
                        id += 1
                self.dataset = yearlama 
            else:
                raise NameError('Select the correct Dataset!')
        print(f'Length of dataset retrieving is.. {len(self.dataset)}')
        self.input_length = input_length
        self.output_length = output_length
        self.ids_to_answers = ids_to_answers

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):
        # continual pretraining
        if 'wmt' or 'WMT' in self.args.dataset:
            if self.model_type == 'GPT2':
                input_ = example_batch['original']
                target_= example_batch['original']
            elif self.model_type == 'T5':
                input_ = example_batch['input']
                target_ = example_batch['output']
                if type(input_)!=str:
                    input_=''
                if type(target_)!=str:
                    target_=''   
        # evaluation
        else: 
            if self.args.dataset == 'invariantlama':
                if self.model_type == 'GPT2':
                    input_pre = example_batch['input']
                    for index, word in enumerate(input_pre.split()):
                        if word == '<extra_id_0>':
                            input_pre = ' '.join(input_pre.split()[:index])
                            break
                    if self.type_path == 'train':
                        input_ = input_pre + ' ' + example_batch['output'] + '.'
                        target_= input_pre + ' ' + example_batch['output'] + '.'
                    else: 
                        input_ = input_pre
                        ground_truth_ = example_batch['output']
                        target_ = input_pre + ' ' + example_batch['output'] + '.'
                elif self.model_type == 'T5':
                    input_ = example_batch['input']
                    target_ = example_batch['output']
            elif self.args.dataset == 'templama': #zzh delete updatelama and newlama, add templama
                input_ =  example_batch['input']
                target_ = example_batch['output']
            else:
                raise Exception('Select the correct dataset!')
        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt") 
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")     
        
        if self.type_path == 'validation' and self.model_type =='GPT2':
            ground_truth = self.tokenizer.batch_encode_plus([str(ground_truth_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")  
        else: 
            ground_truth = None
        
        if (self.args.dataset == 'invariantlama' or self.args.dataset== 'templama'):
            labels = example_batch['id']
        elif (self.args.dataset == 'newlama' or self.args.dataset == 'updatedlama' or self.args.dataset == 'newlama_easy' or self.args.dataset == 'newqa_easy'):
            labels = example_batch['unique_id']
        else:
            labels = None                 
        return source, targets, labels, ground_truth
  
    def __getitem__(self, index):
        source, targets, labels, ground_truth = self.convert_to_features(self.dataset.iloc[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        if labels is not None:
            label_ids = labels
        else:
            label_ids = -1
        
        if ground_truth is not None:
            ground_truth_ids = ground_truth["input_ids"].squeeze()
        else: 
            ground_truth_ids = -1

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "label_ids": label_ids, "ground_truth_ids": ground_truth_ids}
