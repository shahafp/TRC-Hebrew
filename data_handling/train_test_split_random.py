import pandas as pd
from collections import Counter

data = pd.read_csv('../hw_trc_dataset.csv').sample(frac=1)

docs_df = [df for _, df in data.groupby('document_number')]

cat_train = data[:int(0.8 * len(data))]
cat_test = data[len(cat_train):]

split = len(cat_train) / len(data)

train_counter = Counter(cat_train['label'].tolist())
train_counter_pre = {label: count / len(cat_train) for label, count in train_counter.items()}

test_counter = Counter(cat_test['label'].tolist())
test_counter_pre = {label: count / len(cat_test) for label, count in test_counter.items()}

cat_train.to_csv('split_data/train.csv', index=False)
cat_test.to_csv('split_data/test.csv', index=False)
