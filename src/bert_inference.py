from transformers import AutoTokenizer,TFBertModel
from utils import create_bert
import numpy as np
import pandas as pd

labels = pd.read_csv('../input/labels.csv',index_col=0)

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert = TFBertModel.from_pretrained('bert-base-cased')

covid_model = create_bert(bert)

path_to_weights = '../models/bert_final'
covid_model.load_weights(path_to_weights)

while True:
    texts = input(str('input the text: '))
    if texts == '/stop':
        break
    else:
        x_val = tokenizer(
            text=texts,
            add_special_tokens=True,
            max_length=67,
            truncation=True,
            padding='max_length', 
            return_tensors='tf',
            return_token_type_ids = False,
            return_attention_mask = True,
            verbose = True) 
        validation = covid_model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
    print('Bot: ',labels['answer'][np.argmax(validation[0])])

