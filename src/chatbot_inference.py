import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from utils import bag_of_words

from keras.models import load_model
model = load_model('../models/chatbot_model/chatbot_model.h5')
import json


f = open('../input/common_intents.json')
intents = json.load(f)
words = pickle.load(open('../input/words.pkl','rb'))
classes = pickle.load(open('../input/classes.pkl','rb'))

def predict_class(sentence):
    # filter below  threshold predictions
    p = bag_of_words(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    # print(ints)
    tag = ints[0]['intent']
    # print(tag)
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

while True:
    texts = input(str('input the text: '))
    if texts == '/stop':
        break
    else:
        ints = predict_class(texts)
        result = getResponse(ints, intents)
    print('Bot: ', result)