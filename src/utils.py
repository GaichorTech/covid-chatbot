import re
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
import numpy as np

# define a function to clean the text
def clean_text(text):
    """
    This function returns clean lowercase text after removing all characters other than the alphabets
    :param text: str
    :return str
    """
    temp = re.sub('[^a-zA-Z]', ' ', text)
    temp = temp.lower()
    temp = temp.split()
    temp = ' '.join(temp)
    return temp

# define a function to create a uncased bert model fine-tuned archicture
def create_bert(bert):
    """
    This function returns a fine-tuned bert model
    :param bert: pre-trained bert model
    :return model: fine-tuned bert model
    """
    max_len = 67
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
    embeddings = bert(input_ids,attention_mask = input_mask)[0] 
    out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
    out = Dense(500, activation='relu')(out)
    out = tf.keras.layers.Dropout(0.1)(out)
    out = Dense(300,activation = 'relu')(out)
    y = Dense(271,activation = 'softmax')(out)
    model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
    model.layers[2].trainable = True
    return model

# define a function to clean up the input from user and tokenize it for inference
def clean_up_sentence(sentence):
    """
    This function tokenizes sentence using nltk library
    :param sentence: str
    :return sentence_words: list
    """
    # tokenize the pattern - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=True):
    """
    This function represents words in documents in sparse matrix format
    :param sentence: str
    :param words: list
    :param show_details: bool
    :return numpy array
    """
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words - vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))

    
    