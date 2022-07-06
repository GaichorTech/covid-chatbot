import pandas as pd
from utils import clean_text
data = pd.read_excel('../input/QA_COVID_19.xlsx')

df = data.copy(deep=True)

df['Answers'] = df['Answers']+' For more information, please visit '+df['Reference']

df.drop('Reference', axis=1, inplace=True)

df['Questions'] = df['Questions'].apply(clean_text)

df.drop_duplicates(subset='Questions', keep='last',inplace=True)

df.to_csv('../input/bert_train_data.csv',index=False)

print(df.head())
