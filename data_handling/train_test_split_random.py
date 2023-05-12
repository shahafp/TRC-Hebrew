import pandas as pd
from collections import Counter
import random


def split_data(split_number):
    data = pd.read_csv('../TRC_dataset.csv')
    docs_df = [df for _, df in data.groupby('document_number')]
    split = 0
    while (not 0.795 <= split <= 0.805):
        random.shuffle(docs_df)

        train_docs = docs_df[:int(0.812 * len(docs_df))]
        test_docs = docs_df[len(train_docs):]

        cat_train = pd.concat(train_docs, axis=0).sample(frac=1)
        cat_test = pd.concat(test_docs, axis=0)

        split = len(cat_train) / len(data)
        i = 0

    train_counter = Counter(cat_train['label'].tolist())
    train_counter_pre = {label: count / len(cat_train) for label, count in train_counter.items()}

    test_counter = Counter(cat_test['label'].tolist())
    test_counter_pre = {label: count / len(cat_test) for label, count in test_counter.items()}
    assert len(set(cat_train['document_number'].tolist()).intersection(set(cat_test['document_number'].tolist()))) == 0

    def clean_data(data_df):
        label2id = {'BEFORE': 0, 'EQUAL': 2, 'AFTER': 1, 'VAGUE': 3}

        data_df['named_label'] = data_df['label']
        data_df['label'] = data_df['label'].apply(lambda l: label2id[l])
        clean_data_df = data_df[['text', 'label', 'named_label']]
        return clean_data_df

    clean_data(cat_train).to_csv(f'data_splits/split_{split_number}/train.csv', index=False)
    clean_data(cat_test).to_csv(f'data_splits/split_{split_number}/test.csv', index=False)


if __name__ == '__main__':
    split_data(1)
    split_data(2)
    split_data(3)
