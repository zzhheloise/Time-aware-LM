import json
import pandas as pd
import spacy

file_list = ["train.json", "test.json", "val.json"]
templama_data = []
for file_name in file_list:
    file = open(file_name, 'r', encoding='utf-8')
    for line in file.readlines():
        dic = json.loads(line)
        templama_data.append(dic)


nlp = spacy.load("en_core_web_trf")
sentinels=[]
for i in range(100):
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
        input = input + text[previous_end:start_index] + sentinels[sentinel_cnt]
        target = target + sentinels[sentinel_cnt]+" " + word +" "
        previous_end = end_index
        sentinel_cnt+=1
    input = input + text[previous_end:]
    target = target + sentinels[sentinel_cnt]
    return index, text, input, target

templama = []
query_index = 0
for qa in templama_data:
    query_index += 1
    text = 'In '+ qa['date'] + ' ' + qa['query']
    text = text.replace('_X_',qa['most_recent_answer']['name'])
    output = ssm(query_index, text)
    if output: 
        templama.append(output)
pd.DataFrame(templama, columns=['index','original', 'input', 'output']).to_csv('/data/TempLAMA/templama-ssm.csv')