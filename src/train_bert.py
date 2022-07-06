from utils import create_bert
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from transformers import AutoTokenizer,TFBertModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
# read the training data
df = pd.read_csv('../input/bert_train_data.csv')

# separate the features and targets
X_train, y_train = df['Questions'], df['Answers']

# label encode the categories
le = LabelEncoder()
y_train = le.fit_transform(y_train)
labels = pd.DataFrame(columns=[['label','answer']])
labels['label'] = range(len(le.classes_))
labels['answer'] = le.classes_
labels.to_csv('../input/labels.csv')


# # # vectorize the target variable categories
y_train = to_categorical(y_train)

# # # download the pre-trained bert model and preprocessor
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert = TFBertModel.from_pretrained('bert-base-cased')

# # # Tokenize the input (takes some time) 
# # # here tokenizer using from bert-base-cased
x_train = tokenizer(
    text=X_train.tolist(),
    add_special_tokens=True,
    max_length=67,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

input_ids = x_train['input_ids']
attention_mask = x_train['attention_mask']

# # # set adam optimizer
optimizer = Adam(
        learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website 
        epsilon=1e-08,
        decay=0.01,
        clipnorm=1.0)

# # # Set loss and metrics
loss =CategoricalCrossentropy(from_logits = True)
metric = CategoricalAccuracy('balanced_accuracy')

# # # create a bert model using the define function
model = create_bert(bert)
print('model is created!')

# # # Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)
print('model is compiled')

# # # run below code using a gpu for low training time
train_history = model.fit(
    x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,
    y = y_train,
    epochs=30,
    batch_size=25
)
print('model is trained')
# # Save the weights
model.save_weights('../models/bert_final')
print('model is saved')
bert.save_pretrained('../models/bert_base')
print('base bert is saved')
bert.save_pretrained('../models/bert_tokenizer')
print('bert tokenizer is saved')





