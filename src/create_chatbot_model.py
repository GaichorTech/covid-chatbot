import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import random
import numpy
import pandas as pd
import json
from utils import clean_text
import numpy as np
lemmatizer = WordNetLemmatizer()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

# load the covid QnA data into a dataframe
data = pd.read_excel('../input/QA_COVID_19.xlsx')
df = data.copy(deep=True)
# print(df.head())

# load the common intents for building a chatbot
f = open('..\input\common_intents.json')
common_intents = json.load(f)


# preprocess the excel file
df['Answers'] = df['Answers']+' For more information, please visit '+df['Reference']
df.drop('Reference', axis=1, inplace=True)
df['Questions'] = df['Questions'].apply(clean_text)
df.drop_duplicates(subset='Questions', keep='last',inplace=True)

# create a new tag for guessing the covid related question
new_tag = {"tag":"covid_question","patterns":df['Questions'].to_list(),'responses':[],
                                                  'context_set':''}

# add that tag to common_intents json
common_intents['intents'].extend([new_tag])

# create empty lists to store training data for neural network for the chatbot
words=[] # to store every word in the questions
classes = [] # to store type of text that is input from the user
documents = [] # create input and output lists
ignore_letters = ['!', '?', ',', '.']
intents = common_intents.copy()

# fill the lists
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #add documents in the corpus
        documents.append((word, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

# sort classes
classes = sorted(list(set(classes)))

with open('../input/words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('../input/classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
        
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype=object)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
chat_model = Sequential()
chat_model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
chat_model.add(Dropout(0.5))
chat_model.add(Dense(64, activation='relu'))
chat_model.add(Dropout(0.5))
chat_model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
chat_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = chat_model.fit(np.array(train_x), np.array(train_y), epochs=50, batch_size=5, verbose=1)
chat_model.save('../models/chatbot_model/chatbot_model.h5', hist)
print("model created")