from flask import Flask
app = Flask(__name__)
from flask import Flask, jsonify, request 
from transformers import AutoTokenizer,TFBertModel
from utils import create_bert
import numpy as np
import pandas as pd
from keras.models import load_model
import json
import pickle
from utils import bag_of_words
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

path_to_weights = '../models/bert_final'
labels = pd.read_csv('../input/labels.csv',index_col=0)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert = TFBertModel.from_pretrained('bert-base-cased')
covid_model = create_bert(bert)
covid_model.load_weights(path_to_weights)
chatbot = load_model('../models/chatbot_model/chatbot_model.h5')
f = open('../input/common_intents.json')
intents = json.load(f)
words = pickle.load(open('../input/words.pkl','rb'))
classes = pickle.load(open('../input/classes.pkl','rb'))

def predict_class(sentence):
    # filter below  threshold predictions
    p = bag_of_words(sentence, words,show_details=False)
    res = chatbot.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    if tag == 'covid_question':
        result = 'This is a coronavirus related question!'
    else:
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = i['responses'][0]
                break
            # if(i['tag']=='covid_question'):
            #     result = 'This is a coronavirus related question!'
    return result

def predict_bert_resonse(sentence):
    x_val = tokenizer(
            text=sentence,
            add_special_tokens=True,
            max_length=67,
            truncation=True,
            padding='max_length', 
            return_tensors='tf',
            return_token_type_ids = False,
            return_attention_mask = True,
            verbose = True) 
    validation = covid_model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
    # print('Bot: ',labels['answer'][np.argmax(validation[0])])
    result = labels['answer'][np.argmax(validation[0])]
    return result

@app.route('/predict', methods=['POST'])
def chatbot_reply():
    data = request.get_json(force=True)
    sentence = data['sentence']
    print(sentence)
    ints = predict_class(sentence)
    reply = getResponse(ints, intents)
    if reply != 'This is a coronavirus related question!':
        return jsonify(reply)
    else:
        bert_response = predict_bert_resonse(sentence)
        return jsonify(bert_response)

if __name__ == '__main__':
   app.run(debug = True)