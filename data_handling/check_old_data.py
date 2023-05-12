import pandas as pd
from collections import Counter
import random

train_data = pd.read_csv('old_data/train.csv')
test_data = pd.read_csv('old_data/test.csv')

data = pd.concat([train_data,test_data])

docs_df = [df for _, df in data.groupby('document_number')]

random.shuffle(docs_df)

train_docs = docs_df[:int(0.812*len(docs_df))]
test_docs = docs_df[len(train_docs):]

cat_train = pd.concat(train_docs,axis=0).sample(frac=1)
cat_test = pd.concat(test_docs,axis=0)


split = len(cat_train) / len(data)

train_counter = Counter(cat_train['label'].tolist())
train_counter_pre = {label: count / len(cat_train) for label, count in train_counter.items()}

test_counter = Counter(cat_test['label'].tolist())
test_counter_pre = {label: count / len(cat_test) for label, count in test_counter.items()}

cat_train.to_csv('split_data_old/train.csv', index=False)
cat_test.to_csv('split_data_old/test.csv', index=False)
