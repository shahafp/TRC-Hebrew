import pandas as pd
from collections import Counter

data = pd.read_csv('../INCEPTION/inception_round_1/single_tagger_round_1.csv')
docs_df = [df for _, df in data.groupby('document_number')]

train = docs_df[:int(0.87 * len(docs_df))]
test = docs_df[len(train):]

cat_train = pd.concat(train, axis=0)
cat_test = pd.concat(test, axis=0)

split = len(cat_train) / len(data)

train_counter = Counter(cat_train['label'].tolist())
train_counter_pre = {label: count / len(cat_train) for label, count in train_counter.items()}

test_counter = Counter(cat_test['label'].tolist())
test_counter_pre = {label: count / len(cat_test) for label, count in test_counter.items()}

cat_train.to_csv('split_data/train.csv', index=False)
cat_test.to_csv('split_data/test.csv', index=False)
